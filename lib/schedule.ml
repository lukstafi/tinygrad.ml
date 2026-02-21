(** Schedule creation: transforms a lazy UOp graph into executable kernel descriptions.
    Ported from tinygrad/engine/schedule.py (simplified).

    The full tinygrad scheduler runs 15+ graph rewrite passes to identify kernel
    boundaries, assign buffers, and linearize subgraphs. This simplified version
    handles two cases:
    1. Input buffers (from_float_list) — copyin data to device
    2. Elementwise expression graphs over input buffers — lower to a single kernel *)

type exec_item =
  | Kernel of {
      name: string;
      uops: Uop.t list;
      bufs: Device.buffer list;
    }
  | Copyin of {
      buf: Device.buffer;
      data: float array;
    }

(** Global buffer data store: maps buffer UOp id -> float array data.
    Used by from_float_list to associate data with buffer nodes. *)
let buffer_data : (int, float array) Hashtbl.t = Hashtbl.create 64

let store_buffer_data buf_uop_id data =
  Hashtbl.replace buffer_data buf_uop_id data

let get_buffer_data buf_uop_id =
  Hashtbl.find_opt buffer_data buf_uop_id

(** Realized buffer store: maps buffer UOp id -> Device.buffer.
    After a buffer is allocated and filled, it's stored here.
    Also maps output-result UOp id -> Device.buffer for computed results. *)
let realized_buffers : (int, Device.buffer) Hashtbl.t = Hashtbl.create 64

let store_realized buf_uop_id dbuf =
  Hashtbl.replace realized_buffers buf_uop_id dbuf

let get_realized buf_uop_id =
  Hashtbl.find_opt realized_buffers buf_uop_id

(** Monotonic counter for unique kernel names *)
let kernel_counter = ref 0
let fresh_kernel_name () =
  let n = !kernel_counter in
  incr kernel_counter;
  Printf.sprintf "tk_%d" n

(** Reset all scheduler state. Call between independent test runs
    to prevent state leakage. *)
let reset () =
  Hashtbl.clear buffer_data;
  Hashtbl.clear realized_buffers;
  kernel_counter := 0

(** Recursively rebuild a UOp expression, replacing BUFFER→INDEX→LOAD chains
    with PARAM-based loads. Movement ops are passed through (identity indexing
    for elementwise kernels over flat buffers). *)
let rebuild_expr ~buf_id_to_load (root : Uop.t) : Uop.t =
  let cache : (int, Uop.t) Hashtbl.t = Hashtbl.create 64 in
  let rec rebuild (u : Uop.t) : Uop.t =
    match Hashtbl.find_opt cache u.id with
    | Some r -> r
    | None ->
      let result = match u.op with
        | Ops.BUFFER ->
          (match Hashtbl.find_opt buf_id_to_load u.id with
           | Some load -> load
           | None -> u)
        | Ops.INDEX ->
          rebuild (List.hd u.src)
        | Ops.LOAD ->
          rebuild (List.hd u.src)
        | Ops.RESHAPE | Ops.EXPAND | Ops.CONTIGUOUS ->
          rebuild (List.hd u.src)
        | Ops.CAST ->
          let new_src = List.map rebuild u.src in
          Uop.cast u.dtype (List.hd new_src)
        | _ when Ops.Group.is_alu u.op ->
          let new_src = List.map rebuild u.src in
          Uop.alu u.op u.dtype new_src
        | Ops.REDUCE_AXIS ->
          (* If this reduction was already realized, treat it as a buffer load *)
          (match get_realized u.id with
           | Some _dbuf ->
             (match Hashtbl.find_opt buf_id_to_load u.id with
              | Some load -> load
              | None -> u)
           | None -> u)
        | Ops.CONST -> u
        | _ -> u
      in
      Hashtbl.replace cache u.id result;
      result
  in
  rebuild root

(** Lower a lazy UOp expression tree into a concrete kernel UOp graph.
    Given a root UOp that represents a lazy computation (ALU ops over
    BUFFER→INDEX→LOAD chains, or pure CONST expressions), produce:
    - A list of PARAM UOps (one per input buffer, one for output)
    - A kernel body: RANGE loop with LOADs, ALU ops, STORE
    - The list of Device.buffers in param order

    Returns (kernel_uops, param_bufs, output_buf) or None if already realized. *)
let lower_to_kernel ~device ~numel ~kname (root : Uop.t) : (Uop.t list * Device.buffer list * Device.buffer) option =
  (* Check if this root is already realized *)
  match get_realized root.id with
  | Some _ -> None
  | None ->
    (* Collect all realized nodes from the graph (BUFFERs and realized REDUCE_AXIS) *)
    let all_uops = Uop.toposort1 root in
    let input_buffers = List.filter_map (fun (u : Uop.t) ->
      if u.op = Ops.BUFFER || (u.op = Ops.REDUCE_AXIS && get_realized u.id <> None) then
        match get_realized u.id with
        | Some dbuf -> Some (u.id, dbuf)
        | None -> None
      else None
    ) all_uops in

    let n_inputs = List.length input_buffers in
    let input_params = List.mapi (fun i (_buf_id, _dbuf) ->
      Uop.param i (Dtype.ptr Dtype.float32)
    ) input_buffers in

    (* Output param *)
    let out_param = Uop.param n_inputs (Dtype.ptr Dtype.float32) in

    (* Build the loop body *)
    let loop_bound = Uop.const_int Dtype.int32 numel in
    let i = Uop.range loop_bound [0; 0] in

    (* Map realized UOp IDs to their PARAM → INDEX → LOAD chain *)
    let buf_id_to_load : (int, Uop.t) Hashtbl.t = Hashtbl.create 16 in
    List.iteri (fun idx (buf_id, _dbuf) ->
      let param = List.nth input_params idx in
      let indexed = Uop.index param i in
      let loaded = Uop.load indexed in
      Hashtbl.replace buf_id_to_load buf_id loaded
    ) input_buffers;

    let result_val = rebuild_expr ~buf_id_to_load root in

    (* Store to output *)
    let out_indexed = Uop.index out_param i in
    let st = Uop.store out_indexed result_val in
    let end_r = Uop.end_ i in
    let kernel = Uop.sink ~name:kname [st; end_r] in
    let kernel_uops = Uop.toposort1 kernel in

    (* Allocate output buffer *)
    let out_buf = Device.alloc_buffer
      (Device.make_buffer ~device ~size:numel ~dtype:Dtype.float32) in

    let param_bufs = List.map snd input_buffers @ [out_buf] in

    Some (kernel_uops, param_bufs, out_buf)

(** Lower a reduction operation to a kernel.
    For REDUCE_AXIS(expr, ADD/MAX, axes), generates:
    - Full reduction (output_numel=1):
        out[0] = identity; for(i in 0..input_numel) out[0] = op(out[0], expr[i]);
    - Partial reduction (output_numel>1):
        for(o in 0..output_numel) { out[o] = identity;
          for(r in 0..reduce_extent) out[o] = op(out[o], input[stride_fn(o,r)]); }
    [input_numel] is the number of elements in the source before reduction.
    [output_numel] is the number of elements in the output after reduction.
    [reduce_axes] identifies which axes are being reduced (needed for stride computation). *)
let lower_reduce_kernel ~device ~input_numel ~output_numel ~reduce_axes ~input_shape ~kname
    (reduce_op : Ops.t) (source_uop : Uop.t) (root : Uop.t) : (Uop.t list * Device.buffer list * Device.buffer) option =
  match get_realized root.id with
  | Some _ -> None
  | None ->
    let all_uops = Uop.toposort1 source_uop in
    let input_buffers = List.filter_map (fun (u : Uop.t) ->
      if u.op = Ops.BUFFER then
        match get_realized u.id with
        | Some dbuf -> Some (u.id, dbuf)
        | None -> None
      else None
    ) all_uops in

    let n_inputs = List.length input_buffers in
    let input_params = List.mapi (fun i (_buf_id, _dbuf) ->
      Uop.param i (Dtype.ptr Dtype.float32)
    ) input_buffers in
    let out_param = Uop.param n_inputs (Dtype.ptr Dtype.float32) in

    (* Initialize output elements to identity *)
    let identity_val = match reduce_op with
      | Ops.ADD -> 0.0
      | Ops.MAX -> Float.neg_infinity
      | _ -> 0.0
    in

    if output_numel = 1 then begin
      (* Full reduction to a single value: init out[0], loop, accumulate *)
      let zero_idx = Uop.const_int Dtype.int32 0 in
      let out_idx = Uop.index out_param zero_idx in
      let init_store = Uop.store out_idx (Uop.const Dtype.float32 identity_val) in

      let loop_bound = Uop.const_int Dtype.int32 input_numel in
      let i = Uop.range loop_bound [0; 0] in

      let buf_id_to_load : (int, Uop.t) Hashtbl.t = Hashtbl.create 16 in
      List.iteri (fun idx (buf_id, _dbuf) ->
        let param = List.nth input_params idx in
        let indexed = Uop.index param i in
        let loaded = Uop.load indexed in
        Hashtbl.replace buf_id_to_load buf_id loaded
      ) input_buffers;

      let input_val = rebuild_expr ~buf_id_to_load source_uop in

      let loop_out_idx = Uop.index out_param (Uop.sub i i) in
      let cur = Uop.load loop_out_idx in
      let accumulated = Uop.alu reduce_op Dtype.float32 [cur; input_val] in
      let st = Uop.store loop_out_idx accumulated in
      let end_r = Uop.end_ i in
      let kernel = Uop.sink ~name:kname [init_store; st; end_r] in
      let kernel_uops = Uop.toposort1 kernel in

      let out_buf = Device.alloc_buffer
        (Device.make_buffer ~device ~size:output_numel ~dtype:Dtype.float32) in
      let param_bufs = List.map snd input_buffers @ [out_buf] in
      Some (kernel_uops, param_bufs, out_buf)
    end else begin
      (* Partial reduction: output_numel > 1.
         For input_shape [d0, d1, ..., dn] reducing axis k:
         - reduce_extent = d_k
         - inner_numel = product of dims after the reduced axis
         - flat_idx(o, r) = (o / inner_numel) * (reduce_extent * inner_numel)
                          + r * inner_numel + (o mod inner_numel)
         This handles both last-axis (inner_numel=1) and non-last-axis reductions. *)
      let reduce_extent = input_numel / output_numel in

      (* Compute inner_numel: product of dims after all reduced axes.
         For single-axis reduction of axis k in shape [d0..dn]:
           inner_numel = product of d_{k+1} ... d_n *)
      let ndims = List.length input_shape in
      let inner_numel =
        if reduce_axes = [] then 1
        else
          let max_reduce_axis = List.fold_left max 0 reduce_axes in
          let rec prod_from i acc =
            if i >= ndims then acc
            else prod_from (i + 1) (acc * List.nth input_shape i)
          in
          prod_from (max_reduce_axis + 1) 1
      in

      let outer_bound = Uop.const_int Dtype.int32 output_numel in
      let o = Uop.range outer_bound [0; 0] in

      (* Init out[o] = identity *)
      let out_o_idx = Uop.index out_param o in
      let init_store = Uop.store out_o_idx (Uop.const Dtype.float32 identity_val) in

      let inner_bound = Uop.const_int Dtype.int32 reduce_extent in
      let r = Uop.range inner_bound [1; 0] in

      (* Compute flat index into the input buffer.
         flat_idx = (o / inner_numel) * (reduce_extent * inner_numel)
                  + r * inner_numel + (o % inner_numel) *)
      let flat_idx =
        if inner_numel = 1 then
          (* Last-axis reduction: simple o * reduce_extent + r *)
          Uop.add (Uop.mul o (Uop.const_int Dtype.int32 reduce_extent)) r
        else
          let inner_c = Uop.const_int Dtype.int32 inner_numel in
          let stride_c = Uop.const_int Dtype.int32 (reduce_extent * inner_numel) in
          let outer_idx = Uop.idiv o inner_c in
          let inner_offset = Uop.mod_ o inner_c in
          Uop.add (Uop.add (Uop.mul outer_idx stride_c) (Uop.mul r inner_c)) inner_offset
      in

      let buf_id_to_load : (int, Uop.t) Hashtbl.t = Hashtbl.create 16 in
      List.iteri (fun idx (buf_id, _dbuf) ->
        let param = List.nth input_params idx in
        let indexed = Uop.index param flat_idx in
        let loaded = Uop.load indexed in
        Hashtbl.replace buf_id_to_load buf_id loaded
      ) input_buffers;

      let input_val = rebuild_expr ~buf_id_to_load source_uop in

      (* Read current accumulator, apply reduce_op, write back.
         Use (o + (r - r)) so the INDEX depends on the inner loop variable r,
         forcing the LOAD to be placed inside the inner loop during linearization. *)
      let inner_out_idx = Uop.index out_param (Uop.add o (Uop.sub r r)) in
      let cur = Uop.load inner_out_idx in
      let accumulated = Uop.alu reduce_op Dtype.float32 [cur; input_val] in
      let inner_st = Uop.store inner_out_idx accumulated in
      let end_inner = Uop.end_ r in
      let end_outer = Uop.end_ o in
      let kernel = Uop.sink ~name:kname [init_store; inner_st; end_inner; end_outer] in
      let kernel_uops = Uop.toposort1 kernel in

      let out_buf = Device.alloc_buffer
        (Device.make_buffer ~device ~size:output_numel ~dtype:Dtype.float32) in
      let param_bufs = List.map snd input_buffers @ [out_buf] in
      Some (kernel_uops, param_bufs, out_buf)
    end

(** Compute the input numel for a REDUCE_AXIS node by scanning its source
    subgraph for realized buffer sizes. *)
let infer_reduce_input_numel (source_uop : Uop.t) : int =
  let src_uops = Uop.toposort1 source_uop in
  List.fold_left (fun acc (u : Uop.t) ->
    match u.dtype with
    | Dtype.Ptr (_, _, sz) when sz > 0 -> max acc sz
    | _ -> acc
  ) 0 src_uops

(** Create schedule for a UOp root.
    Handles input buffer realization, reduction kernels, and expression computation.
    [output_shape] is the shape of the output tensor.
    Each REDUCE_AXIS node computes its own input_numel from source buffer sizes. *)
let create_schedule ?(device="CPU") ~output_shape (roots : Uop.t list) : exec_item list =
  let numel = Helpers.prod output_shape in
  let items = ref [] in

  List.iter (fun (root : Uop.t) ->
    let all_uops = Uop.toposort1 root in

    (* Step 1: Realize any unrealized input buffers *)
    List.iter (fun (u : Uop.t) ->
      if u.op = Ops.BUFFER && get_realized u.id = None then
        match get_buffer_data u.id with
        | Some data ->
          let size = Array.length data in
          let dtype = match u.dtype with
            | Dtype.Ptr (base, _, _) -> base
            | dt -> dt
          in
          let buf = Device.alloc_buffer
            (Device.make_buffer ~device ~size ~dtype) in
          store_realized u.id buf;
          items := Copyin { buf; data } :: !items
        | None -> ()
    ) all_uops;

    (* Step 2: Schedule any unrealized REDUCE_AXIS nodes.
       Each reduction computes its own input_numel from source buffer sizes.
       The output_shape + reduce_axes give us the stride information needed
       for correct partial-axis reductions.
       This handles both root reductions and nested ones
       (e.g., mean() = sum() * (1/n) where sum is an inner REDUCE_AXIS). *)
    List.iter (fun (u : Uop.t) ->
      if u.op = Ops.REDUCE_AXIS && get_realized u.id = None then begin
        let reduce_op, source_uop, reduce_axes, src_shape = match u.arg with
          | Uop.Axis_arg (axes, op, src_shape) -> (op, List.hd u.src, axes, src_shape)
          | _ -> failwith "REDUCE_AXIS missing Axis_arg"
        in
        (* Compute input_numel: prefer explicit src_shape, fall back to buffer scanning *)
        let reduce_input_numel =
          if src_shape <> [] then Helpers.prod src_shape
          else
            let inferred = infer_reduce_input_numel source_uop in
            if inferred > 0 then inferred else numel
        in
        (* Use explicit src_shape when available, otherwise reconstruct heuristically *)
        let input_shape =
          if src_shape <> [] then src_shape
          else
            let reduce_extent = reduce_input_numel / (max 1 numel) in
            List.mapi (fun i d ->
              if List.mem i reduce_axes then reduce_extent
              else d
            ) output_shape
        in
        let output_numel = numel in
        let kname = fresh_kernel_name () in
        match lower_reduce_kernel ~device ~input_numel:reduce_input_numel ~output_numel ~reduce_axes ~input_shape ~kname reduce_op source_uop u with
        | Some (kernel_uops, param_bufs, out_buf) ->
          items := Kernel { name = kname; uops = kernel_uops; bufs = param_bufs } :: !items;
          store_realized u.id out_buf
        | None -> ()
      end
    ) all_uops;

    (* Step 3: Check if outer computation is needed.
       This covers ALU ops and CONST-only graphs (full/zeros/ones). *)
    let has_alu = List.exists (fun (u : Uop.t) -> Ops.Group.is_alu u.op) all_uops in
    let is_const_graph = (not has_alu)
      && List.exists (fun (u : Uop.t) -> u.op = Ops.CONST) all_uops
      && get_realized root.id = None in
    let is_reduce = root.op = Ops.REDUCE_AXIS in
    if (has_alu || is_const_graph) && not is_reduce then begin
      let kname = fresh_kernel_name () in
      match lower_to_kernel ~device ~numel ~kname root with
      | Some (kernel_uops, param_bufs, out_buf) ->
        items := Kernel { name = kname; uops = kernel_uops; bufs = param_bufs } :: !items;
        store_realized root.id out_buf
      | None -> ()
    end
  ) roots;

  List.rev !items

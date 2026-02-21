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

(** Lower a lazy UOp expression tree into a concrete kernel UOp graph.
    Given a root UOp that represents a lazy computation (ALU ops over
    BUFFER→INDEX→LOAD chains), produce:
    - A list of PARAM UOps (one per input buffer, one for output)
    - A kernel body: RANGE loop with LOADs, ALU ops, STORE
    - The list of Device.buffers in param order

    Returns (kernel_uops, param_bufs, output_buf) or None if already realized. *)
let lower_to_kernel ~device ~numel (root : Uop.t) : (Uop.t list * Device.buffer list * Device.buffer) option =
  (* Collect all BUFFER nodes from the graph *)
  let all_uops = Uop.toposort1 root in
  let input_buffers = List.filter_map (fun (u : Uop.t) ->
    if u.op = Ops.BUFFER then
      match get_realized u.id with
      | Some dbuf -> Some (u.id, dbuf)
      | None -> None
    else None
  ) all_uops in

  if input_buffers = [] then None  (* No realized inputs, nothing to compute *)
  else begin
    (* Check if this root is already realized *)
    match get_realized root.id with
    | Some _ -> None  (* Already done *)
    | None ->
      (* Create PARAM UOps for each input buffer *)
      let n_inputs = List.length input_buffers in
      let input_params = List.mapi (fun i (_buf_id, _dbuf) ->
        Uop.param i (Dtype.ptr Dtype.float32)
      ) input_buffers in

      (* Output param *)
      let out_param = Uop.param n_inputs (Dtype.ptr Dtype.float32) in

      (* Build the loop body *)
      let loop_bound = Uop.const_int Dtype.int32 numel in
      let i = Uop.range loop_bound [0; 0] in

      (* Map buffer UOp IDs to their PARAM → INDEX → LOAD chain *)
      let buf_id_to_load : (int, Uop.t) Hashtbl.t = Hashtbl.create 16 in
      List.iteri (fun idx (buf_id, _dbuf) ->
        let param = List.nth input_params idx in
        let indexed = Uop.index param i in
        let loaded = Uop.load indexed in
        Hashtbl.replace buf_id_to_load buf_id loaded
      ) input_buffers;

      (* Recursively rebuild the expression, replacing BUFFER→INDEX→LOAD chains
         with our new PARAM-based loads *)
      let cache : (int, Uop.t) Hashtbl.t = Hashtbl.create 64 in
      let rec rebuild (u : Uop.t) : Uop.t =
        match Hashtbl.find_opt cache u.id with
        | Some r -> r
        | None ->
          let result = match u.op with
            | Ops.BUFFER ->
              (* This buffer should be replaced by its load *)
              (match Hashtbl.find_opt buf_id_to_load u.id with
               | Some load -> load
               | None -> u)  (* shouldn't happen *)
            | Ops.INDEX ->
              (* Skip INDEX nodes — they're part of the BUFFER→INDEX→LOAD chain *)
              rebuild (List.hd u.src)
            | Ops.LOAD ->
              (* Replace with the rebuilt source (which should be a BUFFER → our load) *)
              rebuild (List.hd u.src)
            | Ops.RESHAPE | Ops.EXPAND | Ops.CONTIGUOUS ->
              (* Movement ops: pass through the source for now
                 (proper implementation would adjust indexing) *)
              rebuild (List.hd u.src)
            | _ when Ops.Group.is_alu u.op ->
              (* ALU op: rebuild sources *)
              let new_src = List.map rebuild u.src in
              Uop.alu u.op u.dtype new_src
            | Ops.CONST -> u  (* Constants pass through *)
            | _ -> u
          in
          Hashtbl.replace cache u.id result;
          result
      in
      let result_val = rebuild root in

      (* Store to output *)
      let out_indexed = Uop.index out_param i in
      let st = Uop.store out_indexed result_val in
      let end_r = Uop.end_ i in
      let kernel = Uop.sink ~name:"tensor_kernel" [st; end_r] in
      let kernel_uops = Uop.toposort1 kernel in

      (* Allocate output buffer *)
      let out_buf = Device.alloc_buffer
        (Device.make_buffer ~device ~size:numel ~dtype:Dtype.float32) in

      (* Build the buffer list in param order: [input0, input1, ..., output] *)
      let param_bufs = List.map snd input_buffers @ [out_buf] in

      Some (kernel_uops, param_bufs, out_buf)
  end

(** Create schedule for a UOp root.
    Handles both input buffer realization and expression computation. *)
let create_schedule ?(device="CPU") (roots : Uop.t list) : exec_item list =
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

    (* Step 2: Check if there are ALU ops that need computation *)
    let has_alu = List.exists (fun (u : Uop.t) -> Ops.Group.is_alu u.op) all_uops in
    if has_alu then begin
      (* Find the numel from shape info or buffer sizes *)
      let numel = List.fold_left (fun acc (u : Uop.t) ->
        if u.op = Ops.BUFFER then
          match get_realized u.id with
          | Some dbuf -> max acc dbuf.size
          | None -> acc
        else acc
      ) 1 all_uops in

      match lower_to_kernel ~device ~numel root with
      | Some (kernel_uops, param_bufs, out_buf) ->
        items := Kernel { name = "tensor_kernel"; uops = kernel_uops; bufs = param_bufs } :: !items;
        store_realized root.id out_buf
      | None -> ()
    end
  ) roots;

  List.rev !items

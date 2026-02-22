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

(** Shape metadata for input buffers: maps buffer UOp id -> tensor shape.
    Stored by from_float_list alongside buffer_data. Used in step-1 copyin
    to populate realized_shapes so downstream kernels can use broadcast_index. *)
let buffer_shapes : (int, int list) Hashtbl.t = Hashtbl.create 64

let store_buffer_shape buf_uop_id shape =
  Hashtbl.replace buffer_shapes buf_uop_id shape

(** Realized buffer store: maps buffer UOp id -> Device.buffer.
    After a buffer is allocated and filled, it's stored here.
    Also maps output-result UOp id -> Device.buffer for computed results. *)
let realized_buffers : (int, Device.buffer) Hashtbl.t = Hashtbl.create 64

(** Shape metadata for realized buffers: maps buffer UOp id -> shape.
    Used for correct broadcast indexing when a realized buffer is smaller
    than the kernel's iteration space. *)
let realized_shapes : (int, int list) Hashtbl.t = Hashtbl.create 64

let store_realized ?(shape=[]) buf_uop_id dbuf =
  Hashtbl.replace realized_buffers buf_uop_id dbuf;
  if shape <> [] then Hashtbl.replace realized_shapes buf_uop_id shape

let get_realized buf_uop_id =
  Hashtbl.find_opt realized_buffers buf_uop_id

let get_realized_shape buf_uop_id =
  Hashtbl.find_opt realized_shapes buf_uop_id

(** Monotonic counter for unique kernel names *)
let kernel_counter = ref 0
let fresh_kernel_name () =
  let n = !kernel_counter in
  incr kernel_counter;
  Printf.sprintf "tk_%d" n

(** Optional hooks called by reset to clear external state tables.
    Reset clears all scheduler state between independent test runs. *)
let reset_hooks : (unit -> unit) list ref = ref []
let register_reset_hook f = reset_hooks := f :: !reset_hooks

let reset () =
  Hashtbl.clear buffer_data;
  Hashtbl.clear buffer_shapes;
  Hashtbl.clear realized_buffers;
  Hashtbl.clear realized_shapes;
  kernel_counter := 0;
  List.iter (fun f -> f ()) !reset_hooks

(** Compute a broadcast-aware index UOp for loading from a buffer with
    [buf_shape] when the kernel iterates over [out_shape].
    Uses stride-based decomposition: for each dimension, if buf_shape[d]=1
    (broadcast), contribute 0; otherwise contribute the coordinate * buf_stride.
    [flat_idx] is the UOp for the flat iteration index over out_shape. *)
let broadcast_index ~(buf_shape : int list) ~(out_shape : int list) (flat_idx : Uop.t) : Uop.t =
  let ndims = List.length out_shape in
  let buf_ndims = List.length buf_shape in
  if buf_ndims > ndims then
    invalid_arg (Printf.sprintf
      "broadcast_index: buf_shape rank %d exceeds out_shape rank %d" buf_ndims ndims);
  (* Validate: each buf dim must be 1 or match the corresponding out dim *)
  let pad_count = ndims - buf_ndims in
  let padded_buf = List.init pad_count (fun _ -> 1) @ buf_shape in
  List.iteri (fun i bd ->
    let od = List.nth out_shape i in
    if bd <> 1 && bd <> od then
      invalid_arg (Printf.sprintf
        "broadcast_index: buf dim %d (size %d) incompatible with out dim %d (size %d)"
        i bd i od)
  ) padded_buf;
  (* Compute strides for out_shape (row-major) *)
  let out_arr = Array.of_list out_shape in
  let out_strides = Array.make ndims 1 in
  for i = ndims - 2 downto 0 do
    out_strides.(i) <- out_strides.(i + 1) * out_arr.(i + 1)
  done;
  (* Compute strides for buf_shape (row-major, skipping broadcast dims) *)
  let buf_arr = Array.of_list padded_buf in
  let buf_strides = Array.make ndims 1 in
  for i = ndims - 2 downto 0 do
    buf_strides.(i) <- buf_strides.(i + 1) * buf_arr.(i + 1)
  done;
  (* Build the index expression *)
  let result = ref (Uop.const_int Dtype.int32 0) in
  let remaining = ref flat_idx in
  for d = 0 to ndims - 1 do
    let dim_size = out_arr.(d) in
    let coord = if d = ndims - 1 then !remaining
      else Uop.idiv !remaining (Uop.const_int Dtype.int32 out_strides.(d)) in
    if d < ndims - 1 then
      remaining := Uop.mod_ !remaining (Uop.const_int Dtype.int32 out_strides.(d));
    (* Only contribute if buf dim is not broadcast *)
    if buf_arr.(d) > 1 then begin
      if buf_strides.(d) = 1 then
        result := Uop.add !result coord
      else
        result := Uop.add !result (Uop.mul coord (Uop.const_int Dtype.int32 buf_strides.(d)));
      (* Wrap coord if buf dim < out dim (shouldn't happen for valid broadcasts) *)
      ignore dim_size
    end
  done;
  !result


(** Transform a flat index through a permutation.
    Given flat_idx over [perm_shape] (the permuted/output shape) and axes
    (the permutation that produced perm_shape from the original shape),
    return a flat index into the original (unpermuted) layout.

    For example, if original shape is [2,3] and axes=[1,0], then perm_shape=[3,2].
    A flat index i into [3,2] is decomposed into coords, inverse-permuted, and
    recomposed into a flat index into [2,3]. *)
let permute_index ~(perm_shape : int list) ~(axes : int list) (flat_idx : Uop.t) : Uop.t =
  let ndims = List.length perm_shape in
  if ndims <= 1 then flat_idx  (* 0-D or 1-D: no-op *)
  else begin
    let perm_arr = Array.of_list perm_shape in
    (* Compute strides for the permuted shape (row-major) *)
    let perm_strides = Array.make ndims 1 in
    for i = ndims - 2 downto 0 do
      perm_strides.(i) <- perm_strides.(i + 1) * perm_arr.(i + 1)
    done;
    (* Decompose flat_idx into coordinates in perm_shape *)
    let coords = Array.make ndims (Uop.const_int Dtype.int32 0) in
    let remaining = ref flat_idx in
    for d = 0 to ndims - 1 do
      if d = ndims - 1 then coords.(d) <- !remaining
      else begin
        coords.(d) <- Uop.idiv !remaining (Uop.const_int Dtype.int32 perm_strides.(d));
        remaining := Uop.mod_ !remaining (Uop.const_int Dtype.int32 perm_strides.(d))
      end
    done;
    (* Compute inverse permutation: inv_axes[axes[i]] = i *)
    let inv_axes = Array.make ndims 0 in
    List.iteri (fun i a -> inv_axes.(a) <- i) axes;
    (* Original shape: orig_shape[i] = perm_shape[inv_axes[i]] ... wait, that's circular.
       Actually: perm_shape[i] = orig_shape[axes[i]], so orig_shape[j] = perm_shape[inv_axes[j]].
       But simpler: coords in orig space: orig_coord[j] = perm_coord[inv_axes[j]]. *)
    let orig_shape = Array.make ndims 0 in
    for j = 0 to ndims - 1 do
      orig_shape.(j) <- perm_arr.(inv_axes.(j))
    done;
    (* Compute strides for original shape *)
    let orig_strides = Array.make ndims 1 in
    for i = ndims - 2 downto 0 do
      orig_strides.(i) <- orig_strides.(i + 1) * orig_shape.(i + 1)
    done;
    (* Recompose: sum of orig_coord[j] * orig_stride[j] *)
    let result = ref (Uop.const_int Dtype.int32 0) in
    for j = 0 to ndims - 1 do
      let orig_coord = coords.(inv_axes.(j)) in
      if orig_strides.(j) = 1 then
        result := Uop.add !result orig_coord
      else
        result := Uop.add !result (Uop.mul orig_coord (Uop.const_int Dtype.int32 orig_strides.(j)))
    done;
    !result
  end

(** Transform a flat index through a SHRINK operation.
    Given flat_idx over [shrunk_shape] (= [hi_i - lo_i for each dim]),
    bounds [(lo0,hi0); ...], and the original (pre-shrink) [orig_shape],
    return a flat index into orig_shape.
    For each coordinate: orig_coord[i] = shrunk_coord[i] + lo_i. *)
let shrink_index ~(shrunk_shape : int list) ~(bounds : (int * int) list)
    ~(orig_shape : int list) (flat_idx : Uop.t) : Uop.t =
  let ndims = List.length shrunk_shape in
  if ndims = 0 then flat_idx
  else begin
    let sh_arr = Array.of_list shrunk_shape in
    let bounds_arr = Array.of_list bounds in
    let orig_arr = Array.of_list orig_shape in
    (* Strides for shrunk shape (to decompose flat_idx) *)
    let sh_strides = Array.make ndims 1 in
    for i = ndims - 2 downto 0 do
      sh_strides.(i) <- sh_strides.(i + 1) * sh_arr.(i + 1)
    done;
    (* Strides for original shape (to recompose) *)
    let orig_strides = Array.make ndims 1 in
    for i = ndims - 2 downto 0 do
      orig_strides.(i) <- orig_strides.(i + 1) * orig_arr.(i + 1)
    done;
    (* Decompose, add offsets, recompose *)
    let result = ref (Uop.const_int Dtype.int32 0) in
    let remaining = ref flat_idx in
    for d = 0 to ndims - 1 do
      let coord = if d = ndims - 1 then !remaining
        else Uop.idiv !remaining (Uop.const_int Dtype.int32 sh_strides.(d)) in
      if d < ndims - 1 then
        remaining := Uop.mod_ !remaining (Uop.const_int Dtype.int32 sh_strides.(d));
      let (lo, _) = bounds_arr.(d) in
      let orig_coord = if lo = 0 then coord
        else Uop.add coord (Uop.const_int Dtype.int32 lo) in
      if orig_strides.(d) = 1 then
        result := Uop.add !result orig_coord
      else
        result := Uop.add !result (Uop.mul orig_coord (Uop.const_int Dtype.int32 orig_strides.(d)))
    done;
    !result
  end

(** Transform a flat index through a PAD operation.
    Given flat_idx over [padded_shape] and padding [(before0,after0); ...],
    return (inner_idx, in_bounds_mask) where:
    - inner_idx is the flat index into the original (pre-pad) shape
    - in_bounds_mask is a UOp that's 1 when the index is in the valid region, 0 in padding
    The caller should use: where(mask, load(inner_idx), 0.0) *)
let pad_index ~(padded_shape : int list) ~(padding : (int * int) list)
    ~(inner_shape : int list) (flat_idx : Uop.t) : Uop.t * Uop.t =
  let ndims = List.length padded_shape in
  if ndims = 0 then (flat_idx, Uop.const Dtype.float32 1.0)
  else begin
    let pad_arr = Array.of_list padded_shape in
    let inner_arr = Array.of_list inner_shape in
    let padding_arr = Array.of_list padding in
    (* Strides for padded shape *)
    let pad_strides = Array.make ndims 1 in
    for i = ndims - 2 downto 0 do
      pad_strides.(i) <- pad_strides.(i + 1) * pad_arr.(i + 1)
    done;
    (* Strides for inner shape *)
    let inner_strides = Array.make ndims 1 in
    for i = ndims - 2 downto 0 do
      inner_strides.(i) <- inner_strides.(i + 1) * inner_arr.(i + 1)
    done;
    (* Decompose padded coords, check bounds, compute inner index *)
    let inner_idx = ref (Uop.const_int Dtype.int32 0) in
    let mask = ref (Uop.const Dtype.float32 1.0) in
    let remaining = ref flat_idx in
    for d = 0 to ndims - 1 do
      let coord = if d = ndims - 1 then !remaining
        else Uop.idiv !remaining (Uop.const_int Dtype.int32 pad_strides.(d)) in
      if d < ndims - 1 then
        remaining := Uop.mod_ !remaining (Uop.const_int Dtype.int32 pad_strides.(d));
      let (before, _after) = padding_arr.(d) in
      (* inner_coord = coord - before *)
      let inner_coord = if before = 0 then coord
        else Uop.sub coord (Uop.const_int Dtype.int32 before) in
      (* Check: 0 <= inner_coord < inner_dim *)
      let in_lo = if before = 0 then Uop.const Dtype.float32 1.0
        else
          (* coord >= before <=> inner_coord >= 0 *)
          (* Use: NOT (inner_coord < 0) = NOT (coord < before) *)
          let cmp = Uop.alu Ops.CMPLT Dtype.float32 [
            Uop.cast Dtype.float32 coord;
            Uop.const Dtype.float32 (Float.of_int before)
          ] in
          Uop.sub (Uop.const Dtype.float32 1.0) cmp
      in
      let in_hi =
        (* inner_coord < inner_dim *)
        let cmp = Uop.alu Ops.CMPLT Dtype.float32 [
          Uop.cast Dtype.float32 inner_coord;
          Uop.const Dtype.float32 (Float.of_int inner_arr.(d))
        ] in
        cmp
      in
      mask := Uop.mul !mask (Uop.mul in_lo in_hi);
      (* Accumulate inner index *)
      if inner_strides.(d) = 1 then
        inner_idx := Uop.add !inner_idx inner_coord
      else
        inner_idx := Uop.add !inner_idx (Uop.mul inner_coord (Uop.const_int Dtype.int32 inner_strides.(d)))
    done;
    (!inner_idx, !mask)
  end

(** Rebuild an expression, replacing buffer references with PARAM-based loads.
    Movement ops are stripped. When the same buffer is accessed through different
    reshape/expand paths, each path gets its own broadcast index (avoiding the
    aliasing bug where a single effective shape was stored per buffer ID).

    [buf_id_to_param] maps realized buffer/reduce UOp IDs to (param_uop, dbuf).
    [loop_idx] is the loop induction variable for indexing.
    [output_shape] is the kernel's output shape for broadcast computation. *)
let rebuild_expr ~buf_id_to_param ~loop_idx ~output_shape (root : Uop.t) : Uop.t =
  (* Cache keyed by UOp ID — but for BUFFER/realized nodes we DON'T cache,
     because the same buffer might be reached via different reshape/expand paths
     that need different broadcast indices. Instead, we cache from the EXPAND
     or RESHAPE node level (which has unique UOp IDs for each path). *)
  let cache : (int, Uop.t) Hashtbl.t = Hashtbl.create 64 in

  (* Generate a load for a realized buffer with the given effective shape context.
     [effective_shape] comes from the RESHAPE above the buffer (if any).
     [cur_idx] is the current flat index (may differ from loop_idx if a PERMUTE
     has been applied upstream).
     [idx_shape] is the coordinate frame matching cur_idx — used for broadcast_index
     decomposition instead of output_shape when they differ (e.g., after PERMUTE). *)
  let make_load buf_id effective_shape cur_idx idx_shape =
    let out_sh = idx_shape in
    let idx_numel = List.fold_left ( * ) 1 out_sh in
    match Hashtbl.find_opt buf_id_to_param buf_id with
    | Some (param, dbuf) ->
      let idx_expr =
        if dbuf.Device.size < idx_numel && dbuf.Device.size > 0 then
          let is_valid bs =
            let bn = List.length bs in
            let on = List.length out_sh in
            bn <= on &&
            let pad = on - bn in
            let padded = List.init pad (fun _ -> 1) @ bs in
            List.for_all2 (fun b o -> b = 1 || b = o) padded out_sh
          in
          let buf_shape =
            match effective_shape with
            | Some s when s <> [] && is_valid s -> Some s
            | _ ->
              match get_realized_shape buf_id with
              | Some s when is_valid s -> Some s
              | _ -> None
          in
          match buf_shape with
          | Some bs -> broadcast_index ~buf_shape:bs ~out_shape:out_sh cur_idx
          | None ->
            if dbuf.Device.size = 1 then Uop.const_int Dtype.int32 0
            else Uop.idiv cur_idx (Uop.const_int Dtype.int32 (idx_numel / dbuf.Device.size))
        else cur_idx
      in
      Uop.load (Uop.index param idx_expr)
    | None -> failwith (Printf.sprintf "rebuild_expr: buffer %d not in param map" buf_id)
  in

  (* Path-dependent ops: the result depends on eff_shape or cur_idx flowing
     down from the root, so the same node may need different lowerings on
     different paths. These must bypass the per-node result cache.
     CAST passes cur_idx to children, so it IS path-dependent when cur_idx
     varies (e.g., after a PERMUTE transform). *)
  let is_path_dependent (op : Ops.t) = match op with
    | Ops.BUFFER | Ops.REDUCE_AXIS | Ops.INDEX | Ops.LOAD
    | Ops.RESHAPE | Ops.EXPAND | Ops.CAST -> true
    | _ when Ops.Group.is_view_wrapper op -> true  (* CONTIGUOUS, PERMUTE, PAD, etc. *)
    | _ -> false
  in

  let rec rebuild (u : Uop.t) ~(eff_shape : int list option) ~(cur_idx : Uop.t)
      ~(idx_shape : int list) : Uop.t =
    (* For non-buffer nodes, check cache *)
    match Hashtbl.find_opt cache u.id with
    | Some r when not (is_path_dependent u.op) -> r
    | _ ->
      let result = match u.op with
        | Ops.BUFFER ->
          if Hashtbl.mem buf_id_to_param u.id then
            make_load u.id eff_shape cur_idx idx_shape
          else u
        | Ops.INDEX ->
          rebuild (List.hd u.src) ~eff_shape ~cur_idx ~idx_shape
        | Ops.LOAD ->
          rebuild (List.hd u.src) ~eff_shape ~cur_idx ~idx_shape
        | Ops.RESHAPE ->
          (* Only set effective shape if not already set by an outer EXPAND.
             The outermost RESHAPE (closest to EXPAND) determines the broadcast shape. *)
          let target_shape = match u.arg with Uop.Shape s -> s | _ -> [] in
          let new_eff = match eff_shape with
            | Some _ -> eff_shape  (* already set by outer context, don't override *)
            | None -> if target_shape <> [] then Some target_shape else None
          in
          rebuild (List.hd u.src) ~eff_shape:new_eff ~cur_idx ~idx_shape
        | Ops.EXPAND ->
          (* The EXPAND's child shape (with 1s for broadcast dims) determines
             the buffer's effective shape. Walk through simple wrappers
             (CONTIGUOUS, CAST, PERMUTE, etc.) to find the nearest RESHAPE. *)
          let rec find_reshape (n : Uop.t) =
            match n.op with
            | Ops.RESHAPE -> (match n.arg with Uop.Shape s -> Some s | _ -> None)
            | _ when Ops.Group.is_view_wrapper n.op && n.src <> [] ->
              find_reshape (List.hd n.src)
            | _ -> None
          in
          let child = List.hd u.src in
          let child_shape = find_reshape child in
          let new_eff = match child_shape with
            | Some s -> Some s  (* use the reshape shape before expand *)
            | None -> eff_shape
          in
          rebuild child ~eff_shape:new_eff ~cur_idx ~idx_shape
        | Ops.PERMUTE ->
          (* Apply index permutation: decompose flat index using the permuted
             (output) shape, inverse-permute coordinates, recompose using
             the original (child) shape.
             Find the PERMUTE's input shape from the child graph:
             - RESHAPE provides its shape directly from arg
             - EXPAND provides its (expanded) shape from arg
             - Other view wrappers (CONTIGUOUS, CAST, PAD, etc.) are walked through
             Note: we must NOT walk through EXPAND since it changes logical shape. *)
          let axes = match u.arg with Uop.Int_list a -> a | _ -> [] in
          let child = List.hd u.src in
          let rec find_child_shape (n : Uop.t) = match n.op with
            | Ops.RESHAPE -> (match n.arg with Uop.Shape s -> Some s | _ -> None)
            | Ops.EXPAND -> (match n.arg with Uop.Shape s -> Some s | _ -> None)
            | Ops.PAD when n.src <> [] ->
              (* PAD changes shape: output[i] = input[i] + before_i + after_i *)
              (match find_child_shape (List.hd n.src), n.arg with
               | Some inner, Uop.Pad_arg padding ->
                 Some (List.map2 (fun d (b, a) -> d + b + a) inner padding)
               | _ -> None)
            | Ops.SHRINK when n.src <> [] ->
              (* SHRINK changes shape: output[i] = hi_i - lo_i *)
              (match n.arg with
               | Uop.Pad_arg bounds ->
                 Some (List.map (fun (lo, hi) -> hi - lo) bounds)
               | _ -> None)
            | Ops.CONTIGUOUS | Ops.CAST | Ops.FLIP
              when n.src <> [] -> find_child_shape (List.hd n.src)
            | _ -> None
          in
          let child_shape_opt = find_child_shape child in
          begin match axes, child_shape_opt with
          | _ :: _, Some child_shape when List.length axes = List.length child_shape ->
            (* perm_shape[i] = child_shape[axes[i]] *)
            let perm_shape = List.map (List.nth child_shape) axes in
            let new_idx = permute_index ~perm_shape ~axes cur_idx in
            (* After permute, the index is now in the child's (unpermuted) coordinate frame *)
            rebuild child ~eff_shape ~cur_idx:new_idx ~idx_shape:child_shape
          | _ ->
            (* No shape info available — pass through (identity permute or 1-D) *)
            rebuild child ~eff_shape ~cur_idx ~idx_shape
          end
        | Ops.SHRINK ->
          (* SHRINK: extract a sub-region. Transform index by adding lower-bound offsets. *)
          let bounds = match u.arg with Uop.Pad_arg b -> b | _ -> [] in
          let child = List.hd u.src in
          if bounds <> [] then begin
            let shrunk_shape = List.map (fun (lo, hi) -> hi - lo) bounds in
            (* Find the child (original, pre-shrink) shape *)
            let rec find_shape (n : Uop.t) = match n.op with
              | Ops.RESHAPE -> (match n.arg with Uop.Shape s -> Some s | _ -> None)
              | Ops.EXPAND -> (match n.arg with Uop.Shape s -> Some s | _ -> None)
              | Ops.CONTIGUOUS | Ops.CAST | Ops.FLIP
                when n.src <> [] -> find_shape (List.hd n.src)
              | _ -> None
            in
            match find_shape child with
            | Some orig_shape when List.length orig_shape = List.length bounds ->
              let new_idx = shrink_index ~shrunk_shape ~bounds ~orig_shape cur_idx in
              rebuild child ~eff_shape ~cur_idx:new_idx ~idx_shape:orig_shape
            | _ ->
              rebuild child ~eff_shape ~cur_idx ~idx_shape
          end else
            rebuild child ~eff_shape ~cur_idx ~idx_shape
        | Ops.PAD ->
          (* PAD: add zero-padding. Decompose index, check bounds, mask with 0 for padding. *)
          let padding = match u.arg with Uop.Pad_arg p -> p | _ -> [] in
          let child = List.hd u.src in
          if padding <> [] then begin
            let inner_shape = match u.arg with
              | Uop.Pad_arg _ ->
                (* Inner shape = padded_shape[i] - before_i - after_i for each dim.
                   But we need the actual inner shape. Find it from the child. *)
                let rec find_shape (n : Uop.t) = match n.op with
                  | Ops.RESHAPE -> (match n.arg with Uop.Shape s -> Some s | _ -> None)
                  | Ops.EXPAND -> (match n.arg with Uop.Shape s -> Some s | _ -> None)
                  | Ops.CONTIGUOUS | Ops.CAST | Ops.FLIP
                    when n.src <> [] -> find_shape (List.hd n.src)
                  | _ -> None
                in
                find_shape child
              | _ -> None
            in
            let padded_shape = idx_shape in  (* current idx_shape IS the padded shape *)
            match inner_shape with
            | Some inner when List.length inner = List.length padding ->
              let (inner_idx, mask) = pad_index ~padded_shape ~padding ~inner_shape:inner cur_idx in
              (* Rebuild child with the inner index *)
              let child_val = rebuild child ~eff_shape ~cur_idx:inner_idx ~idx_shape:inner in
              (* Apply mask: where(mask != 0, child_val, 0.0) *)
              Uop.alu Ops.MUL Dtype.float32 [child_val; mask]
            | _ ->
              rebuild child ~eff_shape ~cur_idx ~idx_shape
          end else
            rebuild child ~eff_shape ~cur_idx ~idx_shape
        | Ops.CONTIGUOUS | Ops.FLIP ->
          rebuild (List.hd u.src) ~eff_shape ~cur_idx ~idx_shape
        | Ops.CAST ->
          let new_src = List.map (fun s -> rebuild s ~eff_shape:None ~cur_idx ~idx_shape) u.src in
          Uop.cast u.dtype (List.hd new_src)
        | _ when Ops.Group.is_alu u.op ->
          let new_src = List.map (fun s -> rebuild s ~eff_shape:None ~cur_idx ~idx_shape) u.src in
          Uop.alu u.op u.dtype new_src
        | Ops.REDUCE_AXIS ->
          (* If this reduction was already realized, treat it as a buffer load *)
          (match get_realized u.id with
           | Some _dbuf ->
             if Hashtbl.mem buf_id_to_param u.id then
               make_load u.id eff_shape cur_idx idx_shape
             else u
           | None -> u)
        | Ops.CONST -> u
        | _ -> u
      in
      (* Cache only nodes whose result does NOT depend on the incoming eff_shape. *)
      if not (is_path_dependent u.op) then
        Hashtbl.replace cache u.id result;
      result
  in
  rebuild root ~eff_shape:None ~cur_idx:loop_idx ~idx_shape:output_shape

(** Lower a lazy UOp expression tree into a concrete kernel UOp graph.
    Given a root UOp that represents a lazy computation (ALU ops over
    BUFFER→INDEX→LOAD chains, or pure CONST expressions), produce:
    - A list of PARAM UOps (one per input buffer, one for output)
    - A kernel body: RANGE loop with LOADs, ALU ops, STORE
    - The list of Device.buffers in param order

    Returns (kernel_uops, param_bufs, output_buf) or None if already realized. *)
let lower_to_kernel ~device ~numel ~output_shape ~kname (root : Uop.t) : (Uop.t list * Device.buffer list * Device.buffer) option =
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

    (* Map realized UOp IDs to their (param, dbuf) pairs.
       rebuild_expr will compute per-path broadcast indices dynamically. *)
    let buf_id_to_param : (int, Uop.t * Device.buffer) Hashtbl.t = Hashtbl.create 16 in
    List.iteri (fun idx (buf_id, dbuf) ->
      let param = List.nth input_params idx in
      Hashtbl.replace buf_id_to_param buf_id (param, dbuf)
    ) input_buffers;

    let result_val = rebuild_expr ~buf_id_to_param ~loop_idx:i
      ~output_shape root in

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

      let buf_id_to_param : (int, Uop.t * Device.buffer) Hashtbl.t = Hashtbl.create 16 in
      List.iteri (fun idx (buf_id, dbuf) ->
        let param = List.nth input_params idx in
        Hashtbl.replace buf_id_to_param buf_id (param, dbuf)
      ) input_buffers;

      let input_val = rebuild_expr ~buf_id_to_param ~loop_idx:i
        ~output_shape:input_shape source_uop in

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
         General approach for arbitrary (possibly non-contiguous) reduce_axes:
         - o iterates over output elements (flat index into non-reduced dims)
         - r iterates over reduce elements (flat index into reduced dims)
         - Decompose o and r into per-axis indices, interleave into full
           multi-dim index, then compute flat input index via strides. *)
      let reduce_extent = input_numel / output_numel in
      let ndims = List.length input_shape in

      (* Compute input strides (row-major) *)
      let input_strides = Array.make ndims 1 in
      for i = ndims - 2 downto 0 do
        input_strides.(i) <- input_strides.(i + 1) * List.nth input_shape (i + 1)
      done;

      (* Separate dims into reduced and non-reduced, preserving order *)
      let is_reduced i = List.mem i reduce_axes in

      (* Check if this is a simple contiguous case (single axis or contiguous
         trailing axes) where we can use the fast path *)
      let sorted_axes = List.sort compare reduce_axes in
      let is_contiguous_trailing =
        sorted_axes <> [] &&
        List.nth sorted_axes (List.length sorted_axes - 1) = ndims - 1 &&
        let first = List.hd sorted_axes in
        List.length sorted_axes = ndims - first
      in
      let is_single_axis = List.length reduce_axes = 1 in

      let outer_bound = Uop.const_int Dtype.int32 output_numel in
      let o = Uop.range outer_bound [0; 0] in

      (* Init out[o] = identity *)
      let out_o_idx = Uop.index out_param o in
      let init_store = Uop.store out_o_idx (Uop.const Dtype.float32 identity_val) in

      let inner_bound = Uop.const_int Dtype.int32 reduce_extent in
      let r = Uop.range inner_bound [1; 0] in

      let flat_idx =
        if is_single_axis || is_contiguous_trailing then begin
          (* Fast path for single-axis or contiguous trailing axes.
             inner_numel = product of dims after all reduced axes. *)
          let max_reduce_axis = List.fold_left max 0 reduce_axes in
          let inner_numel = ref 1 in
          for i = max_reduce_axis + 1 to ndims - 1 do
            inner_numel := !inner_numel * List.nth input_shape i
          done;
          let inner = !inner_numel in
          if inner = 1 then
            Uop.add (Uop.mul o (Uop.const_int Dtype.int32 reduce_extent)) r
          else
            let inner_c = Uop.const_int Dtype.int32 inner in
            let stride_c = Uop.const_int Dtype.int32 (reduce_extent * inner) in
            let outer_idx = Uop.idiv o inner_c in
            let inner_offset = Uop.mod_ o inner_c in
            Uop.add (Uop.add (Uop.mul outer_idx stride_c) (Uop.mul r inner_c)) inner_offset
        end else begin
          (* General path: decompose o and r into per-axis indices, compute
             flat input index from strides.
             Non-reduced dims (from o): extract per-dim index via div/mod
             Reduced dims (from r): extract per-dim index via div/mod *)
          let non_reduced_dims = List.init ndims Fun.id
            |> List.filter (fun i -> not (is_reduced i)) in
          let reduced_dims = List.init ndims Fun.id
            |> List.filter is_reduced in

          (* For each axis, compute its index from o or r *)
          let axis_idx = Array.make ndims (Uop.const_int Dtype.int32 0) in

          (* Decompose o into non-reduced axis indices (row-major within non-reduced) *)
          let remaining_o = ref o in
          let nr_count = List.length non_reduced_dims in
          List.iteri (fun pos ax ->
            let dim_size = List.nth input_shape ax in
            if pos = nr_count - 1 then
              (* Last non-reduced dim: just use remaining *)
              axis_idx.(ax) <- !remaining_o
            else begin
              (* Compute product of remaining non-reduced dims *)
              let tail_prod = ref 1 in
              List.iteri (fun j a ->
                if j > pos then tail_prod := !tail_prod * List.nth input_shape a
              ) non_reduced_dims;
              let tp = Uop.const_int Dtype.int32 !tail_prod in
              axis_idx.(ax) <- Uop.idiv !remaining_o tp;
              remaining_o := Uop.mod_ !remaining_o tp
            end;
            ignore dim_size
          ) non_reduced_dims;

          (* Decompose r into reduced axis indices (row-major within reduced) *)
          let remaining_r = ref r in
          let rd_count = List.length reduced_dims in
          List.iteri (fun pos ax ->
            let dim_size = List.nth input_shape ax in
            if pos = rd_count - 1 then
              axis_idx.(ax) <- !remaining_r
            else begin
              let tail_prod = ref 1 in
              List.iteri (fun j a ->
                if j > pos then tail_prod := !tail_prod * List.nth input_shape a
              ) reduced_dims;
              let tp = Uop.const_int Dtype.int32 !tail_prod in
              axis_idx.(ax) <- Uop.idiv !remaining_r tp;
              remaining_r := Uop.mod_ !remaining_r tp
            end;
            ignore dim_size
          ) reduced_dims;

          (* Compute flat index: sum of axis_idx[i] * stride[i] *)
          let flat = ref (Uop.const_int Dtype.int32 0) in
          for i = 0 to ndims - 1 do
            let stride = input_strides.(i) in
            if stride = 1 then
              flat := Uop.add !flat axis_idx.(i)
            else
              flat := Uop.add !flat (Uop.mul axis_idx.(i) (Uop.const_int Dtype.int32 stride))
          done;
          !flat
        end
      in

      let buf_id_to_param : (int, Uop.t * Device.buffer) Hashtbl.t = Hashtbl.create 16 in
      List.iteri (fun idx (buf_id, dbuf) ->
        let param = List.nth input_params idx in
        Hashtbl.replace buf_id_to_param buf_id (param, dbuf)
      ) input_buffers;

      let input_val = rebuild_expr ~buf_id_to_param ~loop_idx:flat_idx
        ~output_shape:input_shape source_uop in

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

    (* Step 1: Realize any unrealized input buffers.
       When from_float_list stored a shape, we pass it to store_realized
       so downstream kernels can use broadcast_index for this buffer. *)
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
          let shape = match Hashtbl.find_opt buffer_shapes u.id with
            | Some s -> s
            | None -> [size]  (* fallback: flat 1-D shape *)
          in
          store_realized ~shape u.id buf;
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
        (* Compute per-node output_numel from its own shape, not root numel.
           The output shape has 1 in each reduced axis, original size elsewhere. *)
        let output_numel =
          if src_shape <> [] then
            List.fold_left (fun acc (i, d) ->
              acc * (if List.mem i reduce_axes then 1 else d)
            ) 1 (List.mapi (fun i d -> (i, d)) src_shape)
          else numel
        in
        let reduce_out_shape =
          if src_shape <> [] then
            List.mapi (fun i d -> if List.mem i reduce_axes then 1 else d) src_shape
          else [output_numel]
        in
        let kname = fresh_kernel_name () in
        match lower_reduce_kernel ~device ~input_numel:reduce_input_numel ~output_numel ~reduce_axes ~input_shape ~kname reduce_op source_uop u with
        | Some (kernel_uops, param_bufs, out_buf) ->
          items := Kernel { name = kname; uops = kernel_uops; bufs = param_bufs } :: !items;
          store_realized ~shape:reduce_out_shape u.id out_buf
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
      match lower_to_kernel ~device ~numel ~output_shape ~kname root with
      | Some (kernel_uops, param_bufs, out_buf) ->
        items := Kernel { name = kname; uops = kernel_uops; bufs = param_bufs } :: !items;
        store_realized ~shape:output_shape root.id out_buf
      | None -> ()
    end
  ) roots;

  List.rev !items

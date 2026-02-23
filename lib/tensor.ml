(** Tensor: the user-facing API for tinygrad_ml.
    Ported from tinygrad/tensor.py (simplified).

    A Tensor wraps a UOp and tracks shape/dtype. All operations build the lazy
    computation graph; actual computation happens only when realize() is called.

    This is the minimal Tensor surface:
    - Creation: full, zeros, ones, from_float_list
    - Elementwise: +, -, *, /, neg, exp2, log2, sin, sqrt, reciprocal
    - Comparison: <, !=, ==, where
    - Movement: reshape, expand, permute, pad, shrink, flip
    - Reduction: sum, max, mean
    - Type: cast
    - Execution: realize, to_float_list *)

type t = {
  mutable uop: Uop.t;
  mutable lazy_uop: Uop.t option;  (** Original computation graph, preserved across realize *)
  shape: int list;
  dtype: Dtype.t;
  device: string;
  requires_grad: bool;
}

(** Internal buffer counter for unique buffer IDs *)
let next_buf_id = ref 0
let fresh_buf_id () =
  let id = !next_buf_id in
  incr next_buf_id;
  id
let () = Schedule.register_reset_hook (fun () -> next_buf_id := 0)

(** Map from realized BUFFER UOp ID → original computation graph UOp.
    Used by backward to splice computation graphs back in when differentiating
    through realized tensors. *)
let lazy_graph_map : (int, Uop.t) Hashtbl.t = Hashtbl.create 64
let () = Schedule.register_reset_hook (fun () -> Hashtbl.clear lazy_graph_map)

(** Forward reference for realize, used by operations that need intermediate
    materialization (e.g., scaled_dot_product_attention). *)
let realize_ref : (t -> t) ref = ref (fun _ -> failwith "realize not yet initialized")

(** Create a tensor from a pre-built UOp *)
let of_uop ?(device="CPU") ?(requires_grad=false) shape dtype uop =
  { uop; lazy_uop = None; shape; dtype; device; requires_grad }

(** Tensor filled with a constant value *)
let full ?(device="CPU") ?(dtype=Dtype.float32) shape (value : float) =
  let numel = Helpers.prod shape in
  (* Create a CONST UOp, then RESHAPE to desired shape *)
  let c = Uop.const dtype value in
  let expanded = if numel > 1 then
    (* CONST → RESHAPE to (1,...) → EXPAND to shape *)
    let ones = List.map (fun _ -> 1) shape in
    let reshaped = Uop.reshape c ones in
    Uop.expand reshaped shape
  else c in
  of_uop ~device shape dtype expanded

let zeros ?(device="CPU") ?(dtype=Dtype.float32) shape = full ~device ~dtype shape 0.0
let ones ?(device="CPU") ?(dtype=Dtype.float32) shape = full ~device ~dtype shape 1.0

(** Create tensors with same shape/dtype/device as another tensor *)
let full_like (t : t) value = full ~device:t.device ~dtype:t.dtype t.shape value
let zeros_like (t : t) = full_like t 0.0
let ones_like (t : t) = full_like t 1.0

(** Create a tensor from a list of floats *)
let from_float_list ?(device="CPU") ?(dtype=Dtype.float32) shape (data : float list) =
  let numel = Helpers.prod shape in
  if List.length data <> numel then
    failwith (Printf.sprintf "data length %d doesn't match shape %s (numel=%d)"
      (List.length data) (String.concat "x" (List.map string_of_int shape)) numel);
  (* Create a buffer UOp and store the data in the schedule's buffer_data table *)
  let buf_id = fresh_buf_id () in
  let buf_uop = Uop.buffer buf_id (Dtype.ptr ~size:numel dtype) in
  (* Store data and shape for later copyin during realize *)
  Schedule.store_buffer_data buf_uop.id (Array.of_list data);
  Schedule.store_buffer_shape buf_uop.id shape;
  (* Build the graph: BUFFER → INDEX → LOAD → RESHAPE *)
  let idx = Uop.const Dtype.int32 0.0 in
  let indexed = Uop.index buf_uop idx in
  let loaded = Uop.load indexed in
  let reshaped = if List.length shape > 0 then Uop.reshape loaded shape else loaded in
  of_uop ~device shape dtype reshaped

(** Uniform random tensor in [0, 1) *)
let rand ?(device="CPU") ?(dtype=Dtype.float32) shape =
  let n = Helpers.prod shape in
  let data = List.init n (fun _ -> Random.float 1.0) in
  from_float_list ~device ~dtype shape data

(** Normal random tensor (mean=0, std=1) via Box-Muller transform *)
let randn ?(device="CPU") ?(dtype=Dtype.float32) shape =
  let n = Helpers.prod shape in
  let data = List.init n (fun _ ->
    (* Box-Muller: generate normal from two uniforms *)
    let u1 = Random.float 1.0 in
    let u2 = Random.float 1.0 in
    let u1 = if u1 < 1e-10 then 1e-10 else u1 in  (* avoid log(0) *)
    Stdlib.sqrt (-2.0 *. Stdlib.log u1) *. Stdlib.cos (2.0 *. Float.pi *. u2)
  ) in
  from_float_list ~device ~dtype shape data

(** Like-variants using source tensor's shape/dtype/device *)
let rand_like (t : t) = rand ~device:t.device ~dtype:t.dtype t.shape
let randn_like (t : t) = randn ~device:t.device ~dtype:t.dtype t.shape

(** Kaiming uniform initialization: U(-bound, bound) where bound = sqrt(6/fan_in).
    Common for weight initialization in neural networks. *)
let kaiming_uniform ?(device="CPU") ?(dtype=Dtype.float32) ~fan_in shape =
  if fan_in <= 0 then invalid_arg (Printf.sprintf "Tensor.kaiming_uniform: fan_in must be > 0, got %d" fan_in);
  let bound = Stdlib.sqrt (6.0 /. Float.of_int fan_in) in
  let n = Helpers.prod shape in
  let data = List.init n (fun _ -> Random.float (2.0 *. bound) -. bound) in
  from_float_list ~device ~dtype shape data

(** Compute broadcast shape: pad with 1s on left, then max per dim *)
let broadcast_shape s1 s2 =
  let n = max (List.length s1) (List.length s2) in
  let pad s = List.init (n - List.length s) (fun _ -> 1) @ s in
  let s1 = pad s1 and s2 = pad s2 in
  List.map2 (fun a b ->
    if a = b then a
    else if a = 1 then b
    else if b = 1 then a
    else invalid_arg (Printf.sprintf "Tensor.broadcast: incompatible dims %d vs %d" a b)
  ) s1 s2

(** Broadcast a tensor to a target shape (pad with 1s on left + expand).
    Uses UOp-level operations to avoid forward-reference to reshape/expand. *)
let broadcast_to (t : t) target_shape =
  let n = List.length target_shape in
  let m = List.length t.shape in
  if t.shape = target_shape then t
  else
    let padded_shape = List.init (n - m) (fun _ -> 1) @ t.shape in
    let t' = if padded_shape <> t.shape then
      let uop = Uop.reshape t.uop padded_shape in
      { t with uop; shape = padded_shape; lazy_uop = None }
    else t in
    if padded_shape = target_shape then t'
    else
      let uop = Uop.expand t'.uop target_shape in
      { t' with uop; shape = target_shape; lazy_uop = None }

(** Elementwise binary operations with automatic broadcasting *)
let binop op (a : t) (b : t) =
  let out_shape = broadcast_shape a.shape b.shape in
  let a' = broadcast_to a out_shape in
  let b' = broadcast_to b out_shape in
  let result_dtype = match op with
    | Ops.CMPLT | Ops.CMPNE | Ops.CMPEQ -> Dtype.bool
    | _ -> a.dtype
  in
  let uop = Uop.alu op a.dtype [a'.uop; b'.uop] in
  { a' with uop; dtype = result_dtype; lazy_uop = None; shape = out_shape }

(** Elementwise unary operations *)
let unop op (a : t) =
  let uop = Uop.alu op a.dtype [a.uop] in
  { a with uop; lazy_uop = None }

let add a b = binop Ops.ADD a b
let sub a b = binop Ops.SUB a b
let mul a b = binop Ops.MUL a b
let div a b =
  let b_recip = unop Ops.RECIPROCAL b in
  mul a b_recip
let neg_ a = unop Ops.NEG a
let exp2_ a = unop Ops.EXP2 a
let log2_ a = unop Ops.LOG2 a
let sin_ a = unop Ops.SIN a
let sqrt_ a = unop Ops.SQRT a
let reciprocal a = unop Ops.RECIPROCAL a
let trunc_ a = unop Ops.TRUNC a

let lt a b = binop Ops.CMPLT a b
let ne a b = binop Ops.CMPNE a b
let eq a b = binop Ops.CMPEQ a b

let where_ cond t f =
  let out_shape = broadcast_shape cond.shape (broadcast_shape t.shape f.shape) in
  let cond' = broadcast_to cond out_shape in
  let t' = broadcast_to t out_shape in
  let f' = broadcast_to f out_shape in
  let uop = Uop.where_ cond'.uop t'.uop f'.uop in
  { t' with uop; lazy_uop = None; shape = out_shape }

(** ge/le/gt derived from lt: a>=b is !(a<b), i.e. where(a<b, 0, 1) *)
let ge a b =
  let cond = lt a b in
  let one = full ~device:a.device ~dtype:Dtype.bool a.shape 1.0 in
  let zero = full ~device:a.device ~dtype:Dtype.bool a.shape 0.0 in
  where_ cond zero one

let le a b = ge b a
let gt a b = lt b a

(** Scalar constant broadcast to tensor shape *)
let const_like (t : t) (v : float) =
  full ~device:t.device ~dtype:t.dtype t.shape v

(** Natural exponential: exp(x) = exp2(x / ln2) *)
let exp (t : t) =
  let inv_ln2 = const_like t (1.0 /. Float.log 2.0) in
  exp2_ (mul t inv_ln2)

(** Natural logarithm: log(x) = log2(x) * ln2 *)
let log (t : t) =
  let ln2 = const_like t (Float.log 2.0) in
  mul (log2_ t) ln2

(** Cast to a new dtype *)
let cast dtype (t : t) =
  let uop = Uop.cast dtype t.uop in
  { t with uop; dtype; lazy_uop = None }

(** Movement operations *)
let reshape (t : t) new_shape =
  assert (Helpers.prod t.shape = Helpers.prod new_shape);
  let uop = Uop.reshape t.uop new_shape in
  { t with uop; shape = new_shape; lazy_uop = None }

let expand (t : t) new_shape =
  (* each dim must be 1 or match *)
  assert (List.length t.shape = List.length new_shape);
  List.iter2 (fun s n -> assert (s = 1 || s = n)) t.shape new_shape;
  let uop = Uop.expand t.uop new_shape in
  { t with uop; shape = new_shape; lazy_uop = None }

let permute (t : t) axes =
  assert (List.length axes = List.length t.shape);
  let new_shape = List.map (List.nth t.shape) axes in
  let uop = Uop.permute t.uop axes in
  { t with uop; shape = new_shape; lazy_uop = None }

let pad (t : t) padding =
  assert (List.length padding = List.length t.shape);
  let new_shape = List.map2 (fun s (b, a) -> s + b + a) t.shape padding in
  let uop = Uop.pad t.uop padding in
  { t with uop; shape = new_shape; lazy_uop = None }

let shrink (t : t) bounds =
  assert (List.length bounds = List.length t.shape);
  let new_shape = List.map (fun (lo, hi) -> hi - lo) bounds in
  let uop = Uop.shrink t.uop bounds in
  { t with uop; shape = new_shape; lazy_uop = None }

let flip (t : t) axes =
  let uop = Uop.flip t.uop axes in
  { t with uop; lazy_uop = None }

(** Transpose: swap last two dimensions (for 2D+) *)
let transpose (t : t) =
  let n = List.length t.shape in
  if n < 2 then failwith "Tensor.transpose: need at least 2 dimensions";
  let axes = List.init n (fun i ->
    if i = n - 2 then n - 1
    else if i = n - 1 then n - 2
    else i
  ) in
  permute t axes

(** Squeeze: remove dimensions of size 1 *)
let squeeze ?(axes=[]) (t : t) =
  let new_shape = if axes = [] then
    List.filter (fun s -> s <> 1) t.shape
  else
    List.filteri (fun i s ->
      not (List.mem i axes && s = 1)
    ) t.shape
  in
  if new_shape = [] then reshape t [1]  (* keep at least 1 dim *)
  else reshape t new_shape

(** Unsqueeze: insert a dimension of size 1 at the given axis *)
let unsqueeze (t : t) axis =
  let n = List.length t.shape in
  let axis = if axis < 0 then n + 1 + axis else axis in
  if axis < 0 || axis > n then
    invalid_arg (Printf.sprintf "Tensor.unsqueeze: axis %d out of range for %d-D tensor" axis n);
  let new_shape = List.init (n + 1) (fun i ->
    if i < axis then List.nth t.shape i
    else if i = axis then 1
    else List.nth t.shape (i - 1)
  ) in
  reshape t new_shape

(** Flatten: collapse dimensions from start_dim to end_dim into one *)
let flatten ?(start_dim=0) ?(end_dim= -1) (t : t) =
  let n = List.length t.shape in
  let start_dim = if start_dim < 0 then n + start_dim else start_dim in
  let end_dim = if end_dim < 0 then n + end_dim else end_dim in
  if start_dim < 0 || start_dim >= n || end_dim < 0 || end_dim >= n || start_dim > end_dim then
    invalid_arg (Printf.sprintf "Tensor.flatten: invalid dims start=%d end=%d for %d-D tensor"
      start_dim end_dim n);
  let before = List.filteri (fun i _ -> i < start_dim) t.shape in
  let middle = List.filteri (fun i _ -> i >= start_dim && i <= end_dim) t.shape in
  let after = List.filteri (fun i _ -> i > end_dim) t.shape in
  let flat_dim = List.fold_left ( * ) 1 middle in
  reshape t (before @ [flat_dim] @ after)

(** Reduction operations *)
let reduce op (t : t) axes =
  let new_shape = List.mapi (fun i s ->
    if List.mem i axes then 1 else s
  ) t.shape in
  let uop = Uop.reduce_axis ~src_shape:t.shape t.uop op axes in
  { t with uop; shape = new_shape; lazy_uop = None }

let sum ?(axes=[]) (t : t) =
  let axes = if axes = [] then List.init (List.length t.shape) Fun.id else axes in
  reduce Ops.ADD t axes

let max_ ?(axes=[]) (t : t) =
  let axes = if axes = [] then List.init (List.length t.shape) Fun.id else axes in
  reduce Ops.MAX t axes

let mean ?(axes=[]) (t : t) =
  let axes = if axes = [] then List.init (List.length t.shape) Fun.id else axes in
  let s = sum ~axes t in
  let n = List.fold_left (fun acc i -> acc * List.nth t.shape i) 1 axes in
  let n_inv = const_like s (1.0 /. Float.of_int n) in
  mul s n_inv

(** Variance along axes: var(x) = mean((x - mean(x))^2).
    [correction] defaults to 1 (Bessel's correction for unbiased estimate). *)
let var ?(axes=[]) ?(correction=1) (t : t) =
  let axes = if axes = [] then List.init (List.length t.shape) Fun.id else axes in
  let n = List.fold_left (fun acc i -> acc * List.nth t.shape i) 1 axes in
  if n - correction <= 0 then
    invalid_arg (Printf.sprintf "Tensor.var: correction=%d >= sample_count=%d" correction n);
  let m = expand (mean ~axes t) t.shape in
  let diff = sub t m in
  let sq = mul diff diff in
  let s = sum ~axes sq in
  let denom = const_like s (1.0 /. Float.of_int (n - correction)) in
  mul s denom

(** Standard deviation along axes: std(x) = sqrt(var(x)) *)
let std ?(axes=[]) ?(correction=1) (t : t) =
  sqrt_ (var ~axes ~correction t)

(** Concatenate tensors along an axis *)
let cat ?(axis=0) (tensors : t list) =
  match tensors with
  | [] -> failwith "Tensor.cat: empty list"
  | [t] -> t
  | first :: _ ->
    let ndim = List.length first.shape in
    let axis = if axis < 0 then ndim + axis else axis in
    if axis < 0 || axis >= ndim then
      invalid_arg (Printf.sprintf "Tensor.cat: axis %d out of range for %d-D tensors" axis ndim);
    (* Validate: all tensors same shape except along cat axis, same device/dtype *)
    List.iter (fun (t : t) ->
      if List.length t.shape <> ndim then
        failwith "Tensor.cat: all tensors must have same number of dimensions";
      if t.device <> first.device then
        invalid_arg (Printf.sprintf "Tensor.cat: device mismatch (%s vs %s)" first.device t.device);
      if t.dtype <> first.dtype then
        invalid_arg "Tensor.cat: dtype mismatch";
      List.iteri (fun i s ->
        if i <> axis && s <> List.nth first.shape i then
          failwith (Printf.sprintf "Tensor.cat: shape mismatch at dim %d" i)
      ) t.shape
    ) tensors;
    let total_size = List.fold_left (fun acc (t : t) -> acc + List.nth t.shape axis) 0 tensors in
    let out_shape = List.mapi (fun i s -> if i = axis then total_size else s) first.shape in
    (* Pad each tensor to the output shape and add them together *)
    let _offset = ref 0 in
    let padded = List.map (fun (t : t) ->
      let dim_size = List.nth t.shape axis in
      let padding = List.mapi (fun i _s ->
        if i = axis then (!_offset, total_size - !_offset - dim_size)
        else (0, 0)
      ) t.shape in
      _offset := !_offset + dim_size;
      pad t padding
    ) tensors in
    let result = List.fold_left add (List.hd padded) (List.tl padded) in
    { result with shape = out_shape }

(** Detach: stops gradient flow through this tensor.
    The value passes through unchanged, but backward treats it as a leaf. *)
let detach (t : t) =
  let uop = Uop.detach t.uop in
  { t with uop; lazy_uop = None }

(** Softmax along a single axis (numerically stable).
    softmax(x, axis) = exp(x - max(x, axis, keepdim)) / sum(exp(...), axis, keepdim)
    Matches tinygrad's _softmax decomposition. *)
let softmax ?(axis= -1) (t : t) =
  let axis = if axis < 0 then List.length t.shape + axis else axis in
  let m = expand (detach (max_ ~axes:[axis] t)) t.shape in
  let shifted = sub t m in
  let e = exp shifted in
  let s = expand (sum ~axes:[axis] e) t.shape in
  div e s

(** Log-softmax along a single axis (numerically stable).
    log_softmax(x, axis) = (x - max(x, axis)) - log(sum(exp(x - max(x, axis)), axis))
    More numerically stable than log(softmax(x)). *)
let log_softmax ?(axis= -1) (t : t) =
  let axis = if axis < 0 then List.length t.shape + axis else axis in
  let m = expand (detach (max_ ~axes:[axis] t)) t.shape in
  let shifted = sub t m in
  let e = exp shifted in
  let s = expand (sum ~axes:[axis] e) t.shape in
  sub shifted (log s)

(** Cross-entropy loss: -mean(sum(log_softmax(logits) * targets, axis=classes_dim)).
    logits: [batch; classes], targets: [batch; classes] (one-hot or soft labels).
    Returns a scalar loss. Matches tinygrad's cross_entropy with one-hot targets. *)
let cross_entropy ?(axis= -1) (logits : t) (targets : t) =
  if logits.shape <> targets.shape then
    invalid_arg (Printf.sprintf "cross_entropy: logits shape [%s] != targets shape [%s]"
      (String.concat ";" (List.map string_of_int logits.shape))
      (String.concat ";" (List.map string_of_int targets.shape)));
  let ls = log_softmax ~axis logits in
  let per_sample = neg_ (sum ~axes:[if axis < 0 then List.length logits.shape + axis else axis]
                           (mul ls targets)) in
  mean per_sample

(** Mean squared error loss: mean((pred - target)^2).
    Note: requires exact shape match between pred and target (no implicit broadcasting).
    This is intentional — accidental broadcasting in losses can mask shape bugs. *)
let mse_loss (pred : t) (target : t) =
  if pred.shape <> target.shape then
    invalid_arg (Printf.sprintf "Tensor.mse_loss: shape mismatch pred=%s target=%s"
      (String.concat "x" (List.map string_of_int pred.shape))
      (String.concat "x" (List.map string_of_int target.shape)));
  let diff = sub pred target in
  mean (mul diff diff)

(** Layer normalization over the last [n] dimensions.
    layer_norm(x, normalized_shape) = (x - mean) / sqrt(var + eps)
    Optionally scales by [weight] and shifts by [bias]. *)
let layer_norm ?(eps=1e-5) ?weight ?bias (t : t) ~normalized_shape =
  let ndim = List.length t.shape in
  let n_norm = List.length normalized_shape in
  if n_norm > ndim then
    invalid_arg (Printf.sprintf "Tensor.layer_norm: normalized_shape has %d dims but tensor has %d" n_norm ndim);
  (* Validate normalized_shape matches trailing dims *)
  let trailing = List.filteri (fun i _ -> i >= ndim - n_norm) t.shape in
  if trailing <> normalized_shape then
    invalid_arg (Printf.sprintf "Tensor.layer_norm: normalized_shape [%s] doesn't match trailing dims [%s]"
      (String.concat ";" (List.map string_of_int normalized_shape))
      (String.concat ";" (List.map string_of_int trailing)));
  let axes = List.init n_norm (fun i -> ndim - n_norm + i) in
  let m = expand (mean ~axes t) t.shape in
  let diff = sub t m in
  let v = expand (var ~axes ~correction:0 t) t.shape in
  let eps_t = const_like t eps in
  let normalized = div diff (sqrt_ (add v eps_t)) in
  (* Broadcast shape: [1; ...; 1; norm_dim0; norm_dim1; ...] *)
  let bcast_shape = List.init ndim (fun i ->
    if i < ndim - n_norm then 1 else List.nth t.shape i
  ) in
  let scaled = match weight with
    | Some w -> mul normalized (expand (reshape w bcast_shape) t.shape)
    | None -> normalized
  in
  match bias with
  | Some b -> add scaled (expand (reshape b bcast_shape) t.shape)
  | None -> scaled

(** Matrix multiply: supports 2-D and batched (3D+) tensors.
    2-D: a[N,K] @ b[K,M] → [N,M].
    Batched: a[...,N,K] @ b[...,K,M] → [...,N,M] with broadcast-compatible batch dims.
    Implemented via reshape + expand + elementwise mul + sum reduction. *)
let matmul (a : t) (b : t) =
  let a_ndim = List.length a.shape in
  let b_ndim = List.length b.shape in
  if a_ndim < 2 || b_ndim < 2 then
    failwith (Printf.sprintf "matmul: both tensors must be at least 2-D, got [%s] and [%s]"
      (String.concat "," (List.map string_of_int a.shape))
      (String.concat "," (List.map string_of_int b.shape)));
  let k_a = List.nth a.shape (a_ndim - 1) in
  let k_b = List.nth b.shape (b_ndim - 2) in
  if k_a <> k_b then
    failwith (Printf.sprintf "matmul: inner dimensions don't match (%d vs %d)" k_a k_b);
  let k = k_a in
  let n = List.nth a.shape (a_ndim - 2) in
  let m = List.nth b.shape (b_ndim - 1) in
  (* Extract batch dims: everything except last 2 dims *)
  let a_batch = List.filteri (fun i _ -> i < a_ndim - 2) a.shape in
  let b_batch = List.filteri (fun i _ -> i < b_ndim - 2) b.shape in
  (* For 2D case: simple, no batch dims *)
  if a_batch = [] && b_batch = [] then begin
    let a3 = reshape a [n; k; 1] in
    let a_exp = expand a3 [n; k; m] in
    let b3 = reshape b [1; k; m] in
    let b_exp = expand b3 [n; k; m] in
    let prod = mul a_exp b_exp in
    let summed = sum ~axes:[1] prod in
    reshape summed [n; m]
  end else begin
    (* Batched matmul: broadcast batch dims, then do inner product *)
    let max_batch_len = max (List.length a_batch) (List.length b_batch) in
    let pad_front lst target =
      let pad = target - List.length lst in
      List.init pad (fun _ -> 1) @ lst
    in
    let ab = pad_front a_batch max_batch_len in
    let bb = pad_front b_batch max_batch_len in
    let batch_shape = List.map2 (fun a b ->
      if a = b then a
      else if a = 1 then b
      else if b = 1 then a
      else failwith (Printf.sprintf "matmul: batch dims not broadcastable (%d vs %d)" a b)
    ) ab bb in
    (* a: [...batch, N, K] → [...batch, N, K, 1] → expand [...batch, N, K, M] *)
    let a_r = reshape a (ab @ [n; k; 1]) in
    let a_exp = expand a_r (batch_shape @ [n; k; m]) in
    (* b: [...batch, K, M] → [...batch, 1, K, M] → expand [...batch, N, K, M] *)
    let b_r = reshape b (bb @ [1; k; m]) in
    let b_exp = expand b_r (batch_shape @ [n; k; m]) in
    let prod = mul a_exp b_exp in
    let k_axis = max_batch_len + 1 in
    let summed = sum ~axes:[k_axis] prod in
    reshape summed (batch_shape @ [n; m])
  end

(** 2D convolution — forward reference filled in after to_float_list is defined.
    Input [C_in, H, W] * weight [C_out, C_in, KH, KW] → [C_out, OH, OW]. *)
let conv2d_ref : (?stride:int -> ?padding:int -> t -> t -> t) ref =
  ref (fun ?stride:_ ?padding:_ _ _ -> failwith "conv2d not yet initialized")
let conv2d ?(stride=1) ?(padding=0) (input : t) (weight : t) : t =
  !conv2d_ref ~stride ~padding input weight

(** ReLU activation: max(0, x).
    Implemented as where(x > 0, x, 0) using comparison + ternary select. *)
let relu (t : t) =
  let zero = zeros ~device:t.device ~dtype:t.dtype t.shape in
  let cond = lt zero t in  (* 0 < x *)
  where_ cond t zero

(** Sigmoid activation: 1 / (1 + exp(-x)).
    Implemented as exp2(x * (1/ln2)) based formula for numerical stability. *)
let sigmoid (t : t) =
  let one = const_like t 1.0 in
  div one (add one (exp (neg_ t)))

(** Tanh activation: tanh(x) = 2*sigmoid(2x) - 1 *)
let tanh_ (t : t) =
  let two = const_like t 2.0 in
  let one = const_like t 1.0 in
  sub (mul two (sigmoid (mul two t))) one

(** Absolute value: abs(x) = where(x >= 0, x, -x) *)
let abs_ (t : t) =
  let zero = zeros ~device:t.device ~dtype:t.dtype t.shape in
  let cond = lt t zero in  (* x < 0 *)
  where_ cond (neg_ t) t

(** Sign: sign(x) = where(x > 0, 1, where(x < 0, -1, 0)) *)
let sign (t : t) =
  let zero = zeros ~device:t.device ~dtype:t.dtype t.shape in
  let one = const_like t 1.0 in
  let neg_one = const_like t (-1.0) in
  let pos = lt zero t in  (* 0 < x, i.e. x > 0 *)
  let neg = lt t zero in  (* x < 0 *)
  where_ pos one (where_ neg neg_one zero)

(** Clamp: clamp(x, min, max) = max(min, min(x, max_val)) *)
let clamp ?(min_val= Float.neg_infinity) ?(max_val= Float.infinity) (t : t) =
  let result = if max_val < Float.infinity then
    let mx = const_like t max_val in
    let cond = lt mx t in  (* max < x, i.e. x > max *)
    where_ cond mx t
  else t in
  if min_val > Float.neg_infinity then
    let mn = const_like result min_val in
    let cond = lt result mn in  (* x < min *)
    where_ cond mn result
  else result

(** GeLU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    Gaussian Error Linear Unit, used in GPT/BERT-style transformers. *)
let gelu (t : t) =
  let half = const_like t 0.5 in
  let one = const_like t 1.0 in
  let coeff = const_like t 0.044715 in
  let sqrt_2_pi = const_like t 0.7978845608 in  (* sqrt(2/pi) *)
  let x3 = mul t (mul t t) in
  let inner = mul sqrt_2_pi (add t (mul coeff x3)) in
  mul (mul half t) (add one (tanh_ inner))

(** SiLU/Swish activation: x * sigmoid(x) *)
let silu (t : t) = mul t (sigmoid t)

(** ELU activation: elu(x, alpha) = x if x >= 0, alpha*(exp(x)-1) otherwise *)
let elu ?(alpha=1.0) (t : t) =
  let zero = zeros ~device:t.device ~dtype:t.dtype t.shape in
  let one = const_like t 1.0 in
  let a = const_like t alpha in
  let pos = lt zero t in  (* x > 0 *)
  let neg_part = mul a (sub (exp t) one) in
  where_ pos t neg_part

(** Softplus activation: softplus(x) = log(1 + exp(x)) *)
let softplus ?(beta=1.0) (t : t) =
  let b = const_like t beta in
  let one = const_like t 1.0 in
  div (log (add one (exp (mul b t)))) b

(** Mish activation: x * tanh(softplus(x)) *)
let mish (t : t) =
  mul t (tanh_ (softplus t))

let leaky_relu ?(neg_slope=0.01) (t : t) =
  let zero = zeros ~device:t.device ~dtype:t.dtype t.shape in
  let slope = const_like t neg_slope in
  let pos = lt zero t in
  where_ pos t (mul slope t)

(** Element-wise power: base ** exponent.
    Uses exp(exponent * log(abs(base))) with sign correction: preserves the sign
    of the base (negative bases produce negative results). This is exact for
    integer exponents of negative bases but approximate for fractional exponents
    of negative bases (returns -(|base|^exp) rather than NaN/complex). *)
let pow_ (base : t) (exponent : t) =
  let out_shape = broadcast_shape base.shape exponent.shape in
  let base' = broadcast_to base out_shape in
  let exponent' = broadcast_to exponent out_shape in
  let magnitude = exp (mul exponent' (log (abs_ base'))) in
  (* For negative bases with integer exponents: sign = (-1)^exp.
     Detect odd integer exponents: trunc(exp) == exp AND trunc(exp/2)*2 != trunc(exp).
     For non-integer exponents with negative bases, return magnitude (positive). *)
  let zero = zeros ~device:base'.device ~dtype:base'.dtype out_shape in
  let neg_base = lt base' zero in
  let two = const_like exponent' 2.0 in
  let exp_trunc = trunc_ exponent' in
  let is_integer = eq exp_trunc exponent' in
  let half_trunc = trunc_ (div exponent' two) in
  let double_half = mul half_trunc two in
  let is_odd = ne double_half exp_trunc in
  (* Only negate when base < 0 AND exponent is an odd integer *)
  let should_negate = where_ neg_base (where_ is_integer (where_ is_odd
    (full ~device:base'.device ~dtype:Dtype.bool out_shape 1.0)
    (full ~device:base'.device ~dtype:Dtype.bool out_shape 0.0))
    (full ~device:base'.device ~dtype:Dtype.bool out_shape 0.0))
    (full ~device:base'.device ~dtype:Dtype.bool out_shape 0.0) in
  where_ should_negate (neg_ magnitude) magnitude

(** Scalar power: x ** p *)
let pow_scalar (t : t) (p : float) =
  let pv = const_like t p in
  pow_ t pv

(** Element-wise minimum *)
let minimum (a : t) (b : t) =
  let out_shape = broadcast_shape a.shape b.shape in
  let a' = broadcast_to a out_shape in
  let b' = broadcast_to b out_shape in
  let cond = lt a' b' in  (* a < b *)
  where_ cond a' b'

(** Element-wise maximum *)
let maximum (a : t) (b : t) =
  let out_shape = broadcast_shape a.shape b.shape in
  let a' = broadcast_to a out_shape in
  let b' = broadcast_to b out_shape in
  let cond = lt b' a' in  (* b < a, i.e. a > b *)
  where_ cond a' b'

(** Global average pool: reduce spatial dims, keep channels.
    Input [C, H, W] → [C, 1, 1]. *)
let global_avg_pool2d (t : t) =
  if List.length t.shape <> 3 then
    invalid_arg "global_avg_pool2d: input must be 3-D [C,H,W]";
  mean ~axes:[1; 2] t

(** Linspace: [n] evenly spaced values from [start] to [stop] (inclusive) *)
let linspace ?(device="CPU") ?(dtype=Dtype.float32) ~start ~stop (n : int) : t =
  if n < 1 then invalid_arg "Tensor.linspace: n must be >= 1";
  if n = 1 then from_float_list ~device ~dtype [1] [start]
  else
    let step = (stop -. start) /. Float.of_int (n - 1) in
    let data = List.init n (fun i -> start +. step *. Float.of_int i) in
    from_float_list ~device ~dtype [n] data

(** Identity matrix: eye(n) returns [n; n] float tensor *)
let eye ?(device="CPU") ?(dtype=Dtype.float32) (n : int) : t =
  let data = List.init (n * n) (fun i -> if i / n = i mod n then 1.0 else 0.0) in
  from_float_list ~device ~dtype [n; n] data

(** Upper triangular: zeros below the k-th diagonal *)
let triu ?(k=0) (t : t) =
  let ndim = List.length t.shape in
  if ndim < 2 then invalid_arg "Tensor.triu: requires at least 2 dimensions";
  let rows = List.nth t.shape (ndim - 2) in
  let cols = List.nth t.shape (ndim - 1) in
  let mask_data = List.init (rows * cols) (fun idx ->
    let r = idx / cols and c = idx mod cols in
    if c >= r + k then 1.0 else 0.0
  ) in
  let batch = List.filteri (fun i _ -> i < ndim - 2) t.shape in
  let batch_size = Helpers.prod batch in
  let full_data = List.concat (List.init batch_size (fun _ -> mask_data)) in
  let mask = from_float_list ~device:t.device ~dtype:t.dtype t.shape full_data in
  mul t mask

(** Lower triangular: zeros above the k-th diagonal *)
let tril ?(k=0) (t : t) =
  let ndim = List.length t.shape in
  if ndim < 2 then invalid_arg "Tensor.tril: requires at least 2 dimensions";
  let rows = List.nth t.shape (ndim - 2) in
  let cols = List.nth t.shape (ndim - 1) in
  let mask_data = List.init (rows * cols) (fun idx ->
    let r = idx / cols and c = idx mod cols in
    if c <= r + k then 1.0 else 0.0
  ) in
  let batch = List.filteri (fun i _ -> i < ndim - 2) t.shape in
  let batch_size = Helpers.prod batch in
  let full_data = List.concat (List.init batch_size (fun _ -> mask_data)) in
  let mask = from_float_list ~device:t.device ~dtype:t.dtype t.shape full_data in
  mul t mask

(** Split a tensor into chunks along an axis *)
let split ?(axis=0) (t : t) (sizes : int list) : t list =
  let ndim = List.length t.shape in
  let axis = if axis < 0 then ndim + axis else axis in
  let dim_size = List.nth t.shape axis in
  let total = List.fold_left Stdlib.( + ) 0 sizes in
  if total <> dim_size then
    invalid_arg (Printf.sprintf "Tensor.split: sizes sum %d != dim %d" total dim_size);
  let _ = List.fold_left (fun offset size ->
    let bounds = List.mapi (fun i s ->
      if i = axis then (offset, offset + size) else (0, s)
    ) t.shape in
    ignore (shrink t bounds : t);
    offset + size
  ) 0 sizes in
  (* Build chunks via shrink *)
  let _, chunks = List.fold_left (fun (offset, acc) size ->
    let bounds = List.mapi (fun i s ->
      if i = axis then (offset, offset + size) else (0, s)
    ) t.shape in
    (offset + size, shrink t bounds :: acc)
  ) (0, []) sizes in
  List.rev chunks

(** Chunk: split into n roughly-equal-sized chunks *)
let chunk ?(axis=0) (t : t) (n : int) : t list =
  if n <= 0 then invalid_arg (Printf.sprintf "Tensor.chunk: n must be > 0, got %d" n);
  let ndim = List.length t.shape in
  let axis = if axis < 0 then ndim + axis else axis in
  let dim_size = List.nth t.shape axis in
  let base = dim_size / n in
  let remainder = dim_size mod n in
  let sizes = List.init n (fun i -> if i < remainder then base + 1 else base) in
  split ~axis t sizes

(** Stack tensors along a new axis.
    All tensors must have identical shapes. The result has one more dimension. *)
let stack ?(axis=0) (tensors : t list) : t =
  if tensors = [] then invalid_arg "Tensor.stack: empty tensor list";
  let first = List.hd tensors in
  let ndim = List.length first.shape in
  let axis = if axis < 0 then ndim + 1 + axis else axis in
  if axis < 0 || axis > ndim then
    invalid_arg (Printf.sprintf "Tensor.stack: axis %d out of range for %d dims" axis ndim);
  List.iter (fun (t : t) ->
    if t.shape <> first.shape then
      invalid_arg "Tensor.stack: all tensors must have identical shapes"
  ) tensors;
  (* Insert size-1 dimension at axis, then concatenate *)
  let expanded = List.map (fun (t : t) ->
    let new_shape = List.init (ndim + 1) (fun i ->
      if i < axis then List.nth t.shape i
      else if i = axis then 1
      else List.nth t.shape (i - 1)
    ) in
    reshape t new_shape
  ) tensors in
  cat ~axis expanded

(** Binary cross-entropy loss: -mean(target*log(pred) + (1-target)*log(1-pred)).
    Note: requires exact shape match between pred and target (no implicit broadcasting).
    This is intentional — accidental broadcasting in losses can mask shape bugs. *)
let binary_cross_entropy (pred : t) (target : t) =
  if pred.shape <> target.shape then
    invalid_arg (Printf.sprintf "Tensor.binary_cross_entropy: shape mismatch pred=%s target=%s"
      (String.concat "x" (List.map string_of_int pred.shape))
      (String.concat "x" (List.map string_of_int target.shape)));
  let eps_val = 1e-7 in
  let eps = const_like pred eps_val in
  let one = const_like pred 1.0 in
  let clamped = clamp ~min_val:eps_val ~max_val:(1.0 -. eps_val) pred in
  let term1 = mul target (log clamped) in
  let term2 = mul (sub one target) (log (sub (add one eps) clamped)) in
  neg_ (mean (add term1 term2))

(** L1 loss (Mean Absolute Error): mean(|pred - target|). *)
let l1_loss (pred : t) (target : t) =
  if pred.shape <> target.shape then
    invalid_arg "l1_loss: shapes must match";
  mean (abs_ (sub pred target))

(** Huber loss (smooth L1): quadratic for small errors, linear for large errors.
    huber(x) = 0.5*x^2 if |x| <= delta, else delta*(|x| - 0.5*delta).
    More robust to outliers than MSE. *)
let huber_loss ?(delta=1.0) (pred : t) (target : t) =
  if pred.shape <> target.shape then
    invalid_arg "huber_loss: shapes must match";
  if delta <= 0.0 then
    invalid_arg (Printf.sprintf "huber_loss: delta must be > 0, got %f" delta);
  let diff = sub pred target in
  let abs_diff = abs_ diff in
  let delta_t = const_like diff delta in
  let half = const_like diff 0.5 in
  let half_delta = const_like diff (0.5 *. delta) in
  (* quadratic part: 0.5 * diff^2 *)
  let quadratic = mul half (mul diff diff) in
  (* linear part: delta * (|diff| - 0.5 * delta) *)
  let linear = mul delta_t (sub abs_diff half_delta) in
  (* select: |diff| <= delta ? quadratic : linear *)
  let mask = le abs_diff delta_t in
  let per_elem = where_ mask quadratic linear in
  mean per_elem

(** Cross-entropy with label smoothing.
    logits: [batch; num_classes], targets: one-hot [batch; num_classes].
    Smooths targets: y_smooth = (1 - alpha) * y + alpha / num_classes.
    alpha=0.0 is equivalent to standard cross_entropy. *)
let cross_entropy_smooth ?(alpha=0.1) (logits : t) (targets : t) =
  if alpha < 0.0 || alpha > 1.0 then
    invalid_arg (Printf.sprintf "cross_entropy_smooth: alpha must be in [0,1], got %f" alpha);
  if logits.shape <> targets.shape then
    invalid_arg "cross_entropy_smooth: logits and targets must have same shape";
  if List.length logits.shape <> 2 then
    invalid_arg "cross_entropy_smooth: expected 2-D [batch, num_classes]";
  let num_classes = List.nth logits.shape 1 in
  let smooth_factor = const_like targets (alpha /. Float.of_int num_classes) in
  let one_minus_alpha = const_like targets (1.0 -. alpha) in
  let smooth_targets = add (mul one_minus_alpha targets) smooth_factor in
  cross_entropy ~axis:(-1) logits smooth_targets

(** Cosine similarity between two tensors along the last axis.
    Returns a tensor with the last dimension removed (squeezed).
    cos_sim(a, b) = sum(a*b) / (||a|| * ||b|| + eps). *)
let cosine_similarity ?(eps=1e-8) (a : t) (b : t) : t =
  if a.shape <> b.shape then
    invalid_arg "cosine_similarity: shapes must match";
  if List.length a.shape < 1 then
    invalid_arg "cosine_similarity: tensors must have at least 1 dimension";
  let last_ax = List.length a.shape - 1 in
  let dot = sum ~axes:[last_ax] (mul a b) in
  let norm_a = sqrt_ (sum ~axes:[last_ax] (mul a a)) in
  let norm_b = sqrt_ (sum ~axes:[last_ax] (mul b b)) in
  let eps_t = const_like norm_a eps in
  let result = div dot (add (mul norm_a norm_b) eps_t) in
  (* Squeeze the reduced last dimension *)
  let out_shape = List.filteri (fun i _ -> i <> last_ax) result.shape in
  reshape result out_shape

(** KL divergence loss: KL(target || pred).
    pred: logits (will be converted via log_softmax internally).
    target: probability distribution (non-negative, sums to 1 along last axis).
    Returns mean of target * (log(target) - log_softmax(pred)). *)
let kl_div_loss (pred : t) (target : t) =
  if pred.shape <> target.shape then
    invalid_arg "kl_div_loss: shapes must match";
  let eps_t = const_like target 1e-8 in
  let safe_target = add target eps_t in
  let log_pred = log_softmax pred in
  mean (mul target (sub (log safe_target) log_pred))

(** L2-normalize along the given axis. Returns x / (||x||_2 + eps). *)
let normalize ?(axis=(-1)) ?(eps=1e-12) (t : t) : t =
  let ndim = List.length t.shape in
  let ax = if axis < 0 then ndim + axis else axis in
  if ax < 0 || ax >= ndim then
    invalid_arg (Printf.sprintf "normalize: axis %d out of range for %d-D tensor" axis ndim);
  let sq = mul t t in
  let norm_sq = sum ~axes:[ax] sq in
  let eps_t = const_like norm_sq eps in
  let norm = sqrt_ (add norm_sq eps_t) in
  div t norm

(** Scaled dot-product attention: softmax(Q @ K^T / sqrt(d_k)) @ V.
    Q: [seq_q; d_k], K: [seq_k; d_k], V: [seq_k; d_v].
    Optional mask: [seq_q; seq_k] with -inf for masked positions.
    Returns: [seq_q; d_v]. *)
let scaled_dot_product_attention ?mask (q : t) (k : t) (v : t) : t =
  if List.length q.shape <> 2 || List.length k.shape <> 2 || List.length v.shape <> 2 then
    invalid_arg "scaled_dot_product_attention: Q, K, V must be 2-D";
  let d_k = List.nth q.shape 1 in
  let kt = transpose k in
  let raw_scores = !realize_ref (matmul q kt) in  (* [seq_q; seq_k] — materialize *)
  let scale = const_like raw_scores (1.0 /. Stdlib.sqrt (Float.of_int d_k)) in
  let scores = mul raw_scores scale in
  let scores = match mask with
    | Some m -> add scores m
    | None -> scores in
  let attn_weights = !realize_ref (softmax ~axis:(-1) scores) in  (* materialize *)
  matmul attn_weights v

(** Causal (lower-triangular) attention mask for sequence length [n].
    Returns [n; n] tensor with 0 on/below diagonal and -1e9 above. *)
let causal_mask ?(device="CPU") ?(dtype=Dtype.float32) (n : int) : t =
  let data = List.init (n * n) (fun idx ->
    let r = idx / n and c = idx mod n in
    if c <= r then 0.0 else -1e9
  ) in
  from_float_list ~device ~dtype [n; n] data

(** Dropout: randomly zero elements with probability [p] during training.
    Scales remaining values by 1/(1-p) to preserve expected value. *)
let dropout ?(p=0.5) (t : t) =
  if p < 0.0 || p > 1.0 then
    invalid_arg (Printf.sprintf "Tensor.dropout: p must be in [0,1], got %f" p);
  if p <= 0.0 then t
  else if p >= 1.0 then zeros_like t
  else
    let mask = rand ~device:t.device ~dtype:t.dtype t.shape in
    let threshold = const_like t p in
    let keep = lt threshold mask in  (* p < mask, i.e. mask > p → keep *)
    let keep_f = cast t.dtype keep in
    let scale = const_like t (1.0 /. (1.0 -. p)) in
    mul (mul t keep_f) scale

(** Realize: trigger computation.
    This is where lazy evaluation ends and actual work begins.
    1. Schedule the computation graph
    2. Compile and execute kernels
    3. Replace tensor's UOp with a BUFFER reference to the realized data *)
let realize (t : t) =
  (* Each REDUCE_AXIS node in the graph infers its own input_numel from
     source buffer sizes inside create_schedule. The output_shape provides
     stride information for partial-axis reductions. *)
  let schedule = Schedule.create_schedule ~device:t.device ~output_shape:t.shape [t.uop] in
  Realize.run_schedule schedule;
  (* After realization, check if a result buffer was stored for this root UOp.
     If so, replace the tensor's UOp with a BUFFER node pointing to the result,
     so that subsequent operations referencing this tensor can find the data. *)
  (match Schedule.get_realized t.uop.id with
   | Some _buf ->
     let buf_id = fresh_buf_id () in
     let buf_uop = Uop.buffer buf_id (Dtype.ptr ~size:(Helpers.prod t.shape) t.dtype) in
     Schedule.store_realized ~shape:t.shape buf_uop.id _buf;
     (* Preserve the computation graph for backward before replacing with BUFFER *)
     if t.lazy_uop = None then t.lazy_uop <- Some t.uop;
     Hashtbl.replace lazy_graph_map buf_uop.id t.uop;
     t.uop <- buf_uop
   | None -> ());
  t
let () = realize_ref := realize

(** Find the realized Device.buffer for this tensor.
    First checks if the root UOp itself was realized (computed result),
    then falls back to looking for BUFFER nodes (input data). *)
let find_realized_buffer (t : t) : Device.buffer option =
  (* Check if the root UOp was realized (result of a kernel computation) *)
  match Schedule.get_realized t.uop.id with
  | Some buf -> Some buf
  | None ->
    (* Fall back to looking for BUFFER nodes in the graph *)
    let uops = Uop.toposort1 t.uop in
    List.find_map (fun (u : Uop.t) ->
      if u.op = Ops.BUFFER then Schedule.get_realized u.id
      else None
    ) uops

(** Convert to flat float list (triggers realize if needed) *)
let to_float_list (t : t) =
  let _t = realize t in
  match find_realized_buffer t with
  | Some buf ->
    let data = Device.copyout_floats buf in
    Array.to_list data
  | None ->
    failwith (Printf.sprintf
      "Tensor.to_float_list: no realized buffer found for Tensor(shape=[%s], device=%s)"
      (String.concat "," (List.map string_of_int t.shape)) t.device)

(** OCaml operator overloads *)
let ( + ) = add
let ( - ) = sub
let ( * ) = mul
let ( / ) = div
let ( ~- ) = neg_

(** Debug: print tensor info *)
let to_string (t : t) =
  Printf.sprintf "Tensor(shape=[%s], dtype=%s, device=%s)"
    (String.concat ", " (List.map string_of_int t.shape))
    (Dtype.to_string t.dtype)
    t.device

(** Number of elements *)
let numel (t : t) = Helpers.prod t.shape

(** Number of dimensions *)
let ndim (t : t) = List.length t.shape

(** Extract a single float from a scalar or 1-element tensor *)
let item (t : t) : float =
  let n = Helpers.prod t.shape in
  if n <> 1 then
    invalid_arg (Printf.sprintf "Tensor.item: expected scalar, got shape [%s] (numel=%d)"
      (String.concat ";" (List.map string_of_int t.shape)) n);
  List.hd (to_float_list t)

(** Create a 1-D tensor with values [0, 1, ..., n-1] *)
let arange ?(device="CPU") ?(dtype=Dtype.float32) (n : int) : t =
  from_float_list ~device ~dtype [n] (List.init n Float.of_int)

(** Contiguous: marks a tensor for contiguous storage *)
let contiguous (t : t) : t =
  let uop = Uop.create Ops.CONTIGUOUS t.dtype [t.uop] in
  { t with uop; lazy_uop = None }

(** One-hot encoding: indices [batch] → one_hot [batch; num_classes] *)
let one_hot ?device ?(dtype=Dtype.float32) ~num_classes (indices : t) : t =
  (* indices shape must be 1-D *)
  assert (List.length indices.shape = 1);
  let device = match device with Some d -> d | None -> indices.device in
  let batch = List.hd indices.shape in
  (* Create class indices: [0, 1, ..., num_classes-1] expanded to [batch, num_classes] *)
  let cls = arange ~device ~dtype num_classes in
  let cls_exp = expand (reshape cls [1; num_classes]) [batch; num_classes] in
  (* Expand input indices to [batch, num_classes] *)
  let idx_exp = expand (reshape indices [batch; 1]) [batch; num_classes] in
  (* Compare: one_hot[i][j] = 1 if indices[i] == j, else 0 *)
  let mask = eq idx_exp cls_exp in
  cast dtype mask

(** Compute gradients of a scalar loss w.r.t. target tensors.
    [loss] must be a scalar (numel=1) tensor.
    Returns list of (target, gradient_tensor) pairs. *)
let backward (loss : t) (targets : t list) : (t * t) list =
  if Helpers.prod loss.shape <> 1 then
    failwith (Printf.sprintf
      "Tensor.backward: loss must be scalar (numel=1), got shape=[%s] (numel=%d)"
      (String.concat "," (List.map string_of_int loss.shape)) (Helpers.prod loss.shape));
  (* Substitute realized BUFFER nodes with their original computation graphs.
     This allows backward to differentiate through tensors that have been realized
     (e.g., via to_float_list) and then reused in new expressions. *)
  let tensor_uop (t : t) : Uop.t =
    match t.lazy_uop with Some lu -> lu | None ->
    match Hashtbl.find_opt lazy_graph_map t.uop.id with Some lu -> lu | None -> t.uop
  in
  let loss_uop = tensor_uop loss in
  let root_grad = Uop.const loss.dtype 1.0 in
  let target_uops = List.map (fun (t : t) -> tensor_uop t) targets in
  (* Walk the loss graph and splice in lazy computation graphs for any BUFFER
     nodes that were produced by realize. This rebuilds the differentiable graph. *)
  let splice_cache : (int, Uop.t) Hashtbl.t = Hashtbl.create 64 in
  let rec splice (u : Uop.t) : Uop.t =
    match Hashtbl.find_opt splice_cache u.id with
    | Some r -> r
    | None ->
      let result =
        if u.op = Ops.BUFFER then
          match Hashtbl.find_opt lazy_graph_map u.id with
          | Some orig -> splice orig  (* recurse into the original graph *)
          | None -> u
        else
          let new_src = List.map splice u.src in
          if List.for_all2 (fun a b -> a == b) u.src new_src then u
          else Uop.create ~arg:u.arg u.op u.dtype new_src
      in
      Hashtbl.replace splice_cache u.id result;
      result
  in
  let spliced_loss = splice loss_uop in
  let spliced_targets = List.map splice target_uops in
  let grad_pairs = Gradient.compute_gradient spliced_loss root_grad spliced_targets in
  List.map (fun ((target_uop : Uop.t), grad_uop) ->
    let target = List.find (fun (tgt : t) ->
      let tgt_uop = splice (tensor_uop tgt) in
      tgt_uop.Uop.id = target_uop.Uop.id) targets in
    let grad_tensor = of_uop ~device:target.device target.shape target.dtype grad_uop in
    (target, grad_tensor)
  ) grad_pairs

(* conv2d implementation — placed after to_float_list so we can extract weights.
   Note: conv2d extracts weight values to host for scheduling compatibility.
   This means gradients do NOT flow through weights. Use for inference only. *)
(* 2D convolution via host-side computation (forward/inference only).
   Extracts input and weight values, computes convolution in OCaml loops,
   reconstructs result tensor. Gradients do not flow through this operation. *)
let _conv2d_impl ?(stride=1) ?(padding=0) (input : t) (weight : t) : t =
  let ( + ) = Stdlib.( + ) and ( - ) = Stdlib.( - )
  and ( * ) = Stdlib.( * ) and ( / ) = Stdlib.( / ) in
  if stride <= 0 then
    invalid_arg (Printf.sprintf "conv2d: stride must be > 0, got %d" stride);
  if padding < 0 then
    invalid_arg (Printf.sprintf "conv2d: padding must be >= 0, got %d" padding);
  if List.length input.shape <> 3 then
    invalid_arg (Printf.sprintf "conv2d: input must be 3-D [C,H,W], got [%s]"
      (String.concat "," (List.map string_of_int input.shape)));
  if List.length weight.shape <> 4 then
    invalid_arg (Printf.sprintf "conv2d: weight must be 4-D [Cout,Cin,KH,KW], got [%s]"
      (String.concat "," (List.map string_of_int weight.shape)));
  let c_in = List.nth input.shape 0 in
  let ih = List.nth input.shape 1 in
  let iw = List.nth input.shape 2 in
  let c_out = List.nth weight.shape 0 in
  let c_in_w = List.nth weight.shape 1 in
  let kh = List.nth weight.shape 2 in
  let kw = List.nth weight.shape 3 in
  if c_in <> c_in_w then
    invalid_arg (Printf.sprintf "conv2d: input channels %d != weight channels %d" c_in c_in_w);
  let eff_h = ih + 2 * padding - kh in
  let eff_w = iw + 2 * padding - kw in
  if eff_h < 0 || eff_w < 0 then
    invalid_arg "conv2d: kernel larger than padded input, check padding/kernel";
  let oh = eff_h / stride + 1 in
  let ow = eff_w / stride + 1 in
  if oh <= 0 || ow <= 0 then
    invalid_arg "conv2d: output dimensions <= 0, check stride/padding/kernel";
  (* Realize input and weight, extract to host arrays *)
  let input_r = !realize_ref input in
  let weight_r = !realize_ref weight in
  let inp_vals = Array.of_list (to_float_list input_r) in
  let w_vals = Array.of_list (to_float_list weight_r) in
  let padded_h = ih + 2 * padding in
  let padded_w = iw + 2 * padding in
  (* Helper: get padded input value *)
  let get_inp ic h w =
    let h' = h - padding in
    let w' = w - padding in
    if h' < 0 || h' >= ih || w' < 0 || w' >= iw then 0.0
    else inp_vals.((ic * ih + h') * iw + w')
  in
  ignore padded_h; ignore padded_w;
  (* Helper: get weight value *)
  let get_w oc ic ki kj =
    w_vals.(((oc * c_in + ic) * kh + ki) * kw + kj)
  in
  (* Compute output *)
  let out_vals = Array.make (c_out * oh * ow) 0.0 in
  for oc = 0 to c_out - 1 do
    for oi = 0 to oh - 1 do
      for oj = 0 to ow - 1 do
        let sum = ref 0.0 in
        for ic = 0 to c_in - 1 do
          for ki = 0 to kh - 1 do
            for kj = 0 to kw - 1 do
              let hi = oi * stride + ki in
              let wi = oj * stride + kj in
              sum := !sum +. (get_inp ic hi wi) *. (get_w oc ic ki kj)
            done
          done
        done;
        out_vals.((oc * oh + oi) * ow + oj) <- !sum
      done
    done
  done;
  from_float_list ~device:input.device ~dtype:input.dtype [c_out; oh; ow]
    (Array.to_list out_vals)

let () = conv2d_ref := _conv2d_impl

(* 1D convolution via host-side computation (forward/inference only).
   input [C_in, L] → weight [C_out, C_in, K] → [C_out, OL].
   Gradients do not flow through this operation. *)
let conv1d ?(stride=1) ?(padding=0) (input : t) (weight : t) : t =
  let ( + ) = Stdlib.( + ) and ( - ) = Stdlib.( - )
  and ( * ) = Stdlib.( * ) and ( / ) = Stdlib.( / ) in
  if stride <= 0 then
    invalid_arg (Printf.sprintf "conv1d: stride must be > 0, got %d" stride);
  if padding < 0 then
    invalid_arg (Printf.sprintf "conv1d: padding must be >= 0, got %d" padding);
  if List.length input.shape <> 2 then
    invalid_arg (Printf.sprintf "conv1d: input must be 2-D [C,L], got [%s]"
      (String.concat "," (List.map string_of_int input.shape)));
  if List.length weight.shape <> 3 then
    invalid_arg (Printf.sprintf "conv1d: weight must be 3-D [Cout,Cin,K], got [%s]"
      (String.concat "," (List.map string_of_int weight.shape)));
  let c_in = List.nth input.shape 0 in
  let il = List.nth input.shape 1 in
  let c_out = List.nth weight.shape 0 in
  let c_in_w = List.nth weight.shape 1 in
  let kl = List.nth weight.shape 2 in
  if c_in <> c_in_w then
    invalid_arg (Printf.sprintf "conv1d: input channels %d != weight channels %d" c_in c_in_w);
  let effective_l = il + 2 * padding - kl in
  if effective_l < 0 then
    invalid_arg "conv1d: kernel larger than padded input, check padding/kernel";
  let ol = effective_l / stride + 1 in
  if ol <= 0 then
    invalid_arg "conv1d: output length <= 0, check stride/padding/kernel";
  let input_r = !realize_ref input in
  let weight_r = !realize_ref weight in
  let inp_vals = Array.of_list (to_float_list input_r) in
  let w_vals = Array.of_list (to_float_list weight_r) in
  let get_inp ic i =
    let i' = i - padding in
    if i' < 0 || i' >= il then 0.0
    else inp_vals.(ic * il + i')
  in
  let get_w oc ic ki =
    w_vals.((oc * c_in + ic) * kl + ki)
  in
  let out_vals = Array.make (c_out * ol) 0.0 in
  for oc = 0 to c_out - 1 do
    for oi = 0 to ol - 1 do
      let s = ref 0.0 in
      for ic = 0 to c_in - 1 do
        for ki = 0 to kl - 1 do
          let ii = oi * stride + ki in
          s := !s +. (get_inp ic ii) *. (get_w oc ic ki)
        done
      done;
      out_vals.(oc * ol + oi) <- !s
    done
  done;
  from_float_list ~device:input.device ~dtype:input.dtype [c_out; ol]
    (Array.to_list out_vals)

(* 1D max pooling via host-side computation (forward/inference only).
   input [C, L] → [C, OL]. Gradients do not flow through this operation. *)
let max_pool1d ?(stride=0) ?(padding=0) ~kernel_size (input : t) : t =
  let ( + ) = Stdlib.( + ) and ( - ) = Stdlib.( - )
  and ( * ) = Stdlib.( * ) and ( / ) = Stdlib.( / ) in
  if kernel_size <= 0 then
    invalid_arg (Printf.sprintf "max_pool1d: kernel_size must be > 0, got %d" kernel_size);
  if padding < 0 then
    invalid_arg (Printf.sprintf "max_pool1d: padding must be >= 0, got %d" padding);
  let stride = if stride = 0 then kernel_size else stride in
  if stride <= 0 then
    invalid_arg (Printf.sprintf "max_pool1d: stride must be > 0, got %d" stride);
  if List.length input.shape <> 2 then
    invalid_arg (Printf.sprintf "max_pool1d: input must be 2-D [C,L], got [%s]"
      (String.concat "," (List.map string_of_int input.shape)));
  let c = List.nth input.shape 0 in
  let il = List.nth input.shape 1 in
  let eff_l = il + 2 * padding - kernel_size in
  if eff_l < 0 then
    invalid_arg "max_pool1d: kernel larger than padded input";
  let ol = eff_l / stride + 1 in
  let input_r = !realize_ref input in
  let inp_vals = Array.of_list (to_float_list input_r) in
  let get_inp ci i =
    let i' = i - padding in
    if i' < 0 || i' >= il then Float.neg_infinity
    else inp_vals.(ci * il + i')
  in
  let out_vals = Array.make (c * ol) Float.neg_infinity in
  for ci = 0 to c - 1 do
    for oi = 0 to ol - 1 do
      let best = ref Float.neg_infinity in
      for ki = 0 to kernel_size - 1 do
        let ii = oi * stride + ki in
        let v = get_inp ci ii in
        if v > !best then best := v
      done;
      out_vals.(ci * ol + oi) <- !best
    done
  done;
  from_float_list ~device:input.device ~dtype:input.dtype [c; ol]
    (Array.to_list out_vals)

(* 1D average pooling via host-side computation (forward/inference only).
   input [C, L] → [C, OL]. Gradients do not flow through this operation. *)
let avg_pool1d ?(stride=0) ?(padding=0) ~kernel_size (input : t) : t =
  let ( + ) = Stdlib.( + ) and ( - ) = Stdlib.( - )
  and ( * ) = Stdlib.( * ) and ( / ) = Stdlib.( / ) in
  if kernel_size <= 0 then
    invalid_arg (Printf.sprintf "avg_pool1d: kernel_size must be > 0, got %d" kernel_size);
  if padding < 0 then
    invalid_arg (Printf.sprintf "avg_pool1d: padding must be >= 0, got %d" padding);
  let stride = if stride = 0 then kernel_size else stride in
  if stride <= 0 then
    invalid_arg (Printf.sprintf "avg_pool1d: stride must be > 0, got %d" stride);
  if List.length input.shape <> 2 then
    invalid_arg (Printf.sprintf "avg_pool1d: input must be 2-D [C,L], got [%s]"
      (String.concat "," (List.map string_of_int input.shape)));
  let c = List.nth input.shape 0 in
  let il = List.nth input.shape 1 in
  let eff_l = il + 2 * padding - kernel_size in
  if eff_l < 0 then
    invalid_arg "avg_pool1d: kernel larger than padded input";
  let ol = eff_l / stride + 1 in
  let input_r = !realize_ref input in
  let inp_vals = Array.of_list (to_float_list input_r) in
  let get_inp ci i =
    let i' = i - padding in
    if i' < 0 || i' >= il then 0.0
    else inp_vals.(ci * il + i')
  in
  let fk = Float.of_int kernel_size in
  let out_vals = Array.make (c * ol) 0.0 in
  for ci = 0 to c - 1 do
    for oi = 0 to ol - 1 do
      let s = ref 0.0 in
      for ki = 0 to kernel_size - 1 do
        let ii = oi * stride + ki in
        s := !s +. get_inp ci ii
      done;
      out_vals.(ci * ol + oi) <- !s /. fk
    done
  done;
  from_float_list ~device:input.device ~dtype:input.dtype [c; ol]
    (Array.to_list out_vals)

(* 2D max pooling via host-side computation (forward/inference only).
   input [C, H, W] → [C, OH, OW]. Gradients do not flow through this operation. *)
let max_pool2d ?(stride=0) ?(padding=0) ~kernel_size (input : t) : t =
  let ( + ) = Stdlib.( + ) and ( - ) = Stdlib.( - )
  and ( * ) = Stdlib.( * ) and ( / ) = Stdlib.( / ) in
  if kernel_size <= 0 then
    invalid_arg (Printf.sprintf "max_pool2d: kernel_size must be > 0, got %d" kernel_size);
  if padding < 0 then
    invalid_arg (Printf.sprintf "max_pool2d: padding must be >= 0, got %d" padding);
  if List.length input.shape <> 3 then
    invalid_arg (Printf.sprintf "max_pool2d: input must be 3-D [C,H,W], got [%s]"
      (String.concat "," (List.map string_of_int input.shape)));
  let stride = if stride = 0 then kernel_size else stride in
  if stride <= 0 then
    invalid_arg (Printf.sprintf "max_pool2d: stride must be > 0, got %d" stride);
  let c = List.nth input.shape 0 in
  let ih = List.nth input.shape 1 in
  let iw = List.nth input.shape 2 in
  let oh = (ih + 2 * padding - kernel_size) / stride + 1 in
  let ow = (iw + 2 * padding - kernel_size) / stride + 1 in
  if oh <= 0 || ow <= 0 then
    invalid_arg "max_pool2d: output dimensions <= 0";
  let input_r = !realize_ref input in
  let inp = if padding > 0 then
    pad input_r [(0, 0); (padding, padding); (padding, padding)]
  else input_r in
  let out_vals = Array.make (c * oh * ow) 0.0 in
  let inp_vals = Array.of_list (to_float_list inp) in
  let inp_h = List.nth inp.shape 1 in
  let inp_w = List.nth inp.shape 2 in
  let get_inp ic h_idx w_idx =
    inp_vals.((ic * inp_h + h_idx) * inp_w + w_idx)
  in
  for ic = 0 to c - 1 do
    for oi = 0 to oh - 1 do
      for oj = 0 to ow - 1 do
        let max_val = ref neg_infinity in
        for ki = 0 to kernel_size - 1 do
          for kj = 0 to kernel_size - 1 do
            let hi = oi * stride + ki in
            let wi = oj * stride + kj in
            let v = get_inp ic hi wi in
            if v > !max_val then max_val := v
          done
        done;
        out_vals.((ic * oh + oi) * ow + oj) <- !max_val
      done
    done
  done;
  from_float_list ~device:input.device ~dtype:input.dtype [c; oh; ow]
    (Array.to_list out_vals)

(* 2D average pooling via host-side computation (forward/inference only).
   input [C, H, W] → [C, OH, OW]. Gradients do not flow through this operation. *)
let avg_pool2d ?(stride=0) ?(padding=0) ~kernel_size (input : t) : t =
  let ( + ) = Stdlib.( + ) and ( - ) = Stdlib.( - )
  and ( * ) = Stdlib.( * ) and ( / ) = Stdlib.( / ) in
  if kernel_size <= 0 then
    invalid_arg (Printf.sprintf "avg_pool2d: kernel_size must be > 0, got %d" kernel_size);
  if padding < 0 then
    invalid_arg (Printf.sprintf "avg_pool2d: padding must be >= 0, got %d" padding);
  if List.length input.shape <> 3 then
    invalid_arg (Printf.sprintf "avg_pool2d: input must be 3-D [C,H,W], got [%s]"
      (String.concat "," (List.map string_of_int input.shape)));
  let stride = if stride = 0 then kernel_size else stride in
  if stride <= 0 then
    invalid_arg (Printf.sprintf "avg_pool2d: stride must be > 0, got %d" stride);
  let c = List.nth input.shape 0 in
  let ih = List.nth input.shape 1 in
  let iw = List.nth input.shape 2 in
  let oh = (ih + 2 * padding - kernel_size) / stride + 1 in
  let ow = (iw + 2 * padding - kernel_size) / stride + 1 in
  if oh <= 0 || ow <= 0 then
    invalid_arg "avg_pool2d: output dimensions <= 0";
  let input_r = !realize_ref input in
  let inp = if padding > 0 then
    pad input_r [(0, 0); (padding, padding); (padding, padding)]
  else input_r in
  let k_count = Float.of_int (kernel_size * kernel_size) in
  let out_vals = Array.make (c * oh * ow) 0.0 in
  let inp_vals = Array.of_list (to_float_list inp) in
  let inp_h = List.nth inp.shape 1 in
  let inp_w = List.nth inp.shape 2 in
  let get_inp ic h_idx w_idx =
    inp_vals.((ic * inp_h + h_idx) * inp_w + w_idx)
  in
  for ic = 0 to c - 1 do
    for oi = 0 to oh - 1 do
      for oj = 0 to ow - 1 do
        let sum_val = ref 0.0 in
        for ki = 0 to kernel_size - 1 do
          for kj = 0 to kernel_size - 1 do
            let hi = oi * stride + ki in
            let wi = oj * stride + kj in
            sum_val := !sum_val +. get_inp ic hi wi
          done
        done;
        out_vals.((ic * oh + oi) * ow + oj) <- !sum_val /. k_count
      done
    done
  done;
  from_float_list ~device:input.device ~dtype:input.dtype [c; oh; ow]
    (Array.to_list out_vals)

(** Argmax along a given axis. Returns integer indices as float tensor.
    Input shape [...; axis_dim; ...] → output with axis dimension removed.
    Host-side operation (no gradient flow). *)
let argmax ?(axis=(-1)) (t : t) : t =
  let ( + ) = Stdlib.( + ) and ( - ) = Stdlib.( - ) and ( * ) = Stdlib.( * ) in
  let ndim = List.length t.shape in
  let ax = if axis < 0 then ndim + axis else axis in
  if ax < 0 || ax >= ndim then
    invalid_arg (Printf.sprintf "argmax: axis %d out of range for %d-D tensor" axis ndim);
  let _t = !realize_ref t in
  let vals = Array.of_list (to_float_list _t) in
  let axis_dim = List.nth t.shape ax in
  let outer_size = ref 1 in
  for i = 0 to ax - 1 do outer_size := !outer_size * List.nth t.shape i done;
  let inner_size = ref 1 in
  for i = ax + 1 to ndim - 1 do inner_size := !inner_size * List.nth t.shape i done;
  let out_size = !outer_size * !inner_size in
  let result = Array.make out_size 0.0 in
  for o = 0 to !outer_size - 1 do
    for i = 0 to !inner_size - 1 do
      let best_idx = ref 0 in
      let best_val = ref neg_infinity in
      for a = 0 to axis_dim - 1 do
        let flat = (o * axis_dim + a) * !inner_size + i in
        if vals.(flat) > !best_val then begin
          best_val := vals.(flat);
          best_idx := a
        end
      done;
      result.(o * !inner_size + i) <- Float.of_int !best_idx
    done
  done;
  let out_shape = List.filteri (fun i _ -> i <> ax) t.shape in
  from_float_list ~device:t.device ~dtype:t.dtype out_shape (Array.to_list result)

(** Argmin along a given axis. Returns integer indices as float tensor.
    Host-side operation (no gradient flow). *)
let argmin ?(axis=(-1)) (t : t) : t =
  let ( + ) = Stdlib.( + ) and ( - ) = Stdlib.( - ) and ( * ) = Stdlib.( * ) in
  let ndim = List.length t.shape in
  let ax = if axis < 0 then ndim + axis else axis in
  if ax < 0 || ax >= ndim then
    invalid_arg (Printf.sprintf "argmin: axis %d out of range for %d-D tensor" axis ndim);
  let _t = !realize_ref t in
  let vals = Array.of_list (to_float_list _t) in
  let axis_dim = List.nth t.shape ax in
  let outer_size = ref 1 in
  for i = 0 to ax - 1 do outer_size := !outer_size * List.nth t.shape i done;
  let inner_size = ref 1 in
  for i = ax + 1 to ndim - 1 do inner_size := !inner_size * List.nth t.shape i done;
  let out_size = !outer_size * !inner_size in
  let result = Array.make out_size 0.0 in
  for o = 0 to !outer_size - 1 do
    for i = 0 to !inner_size - 1 do
      let best_idx = ref 0 in
      let best_val = ref infinity in
      for a = 0 to axis_dim - 1 do
        let flat = (o * axis_dim + a) * !inner_size + i in
        if vals.(flat) < !best_val then begin
          best_val := vals.(flat);
          best_idx := a
        end
      done;
      result.(o * !inner_size + i) <- Float.of_int !best_idx
    done
  done;
  let out_shape = List.filteri (fun i _ -> i <> ax) t.shape in
  from_float_list ~device:t.device ~dtype:t.dtype out_shape (Array.to_list result)

(** Top-k values and indices along a given axis.
    Returns (values, indices) tensors, both with the axis dimension replaced by k.
    Values are sorted in descending order.
    Host-side operation (no gradient flow). *)
let topk ?(axis=(-1)) ~k (t : t) : t * t =
  let ( + ) = Stdlib.( + ) and ( - ) = Stdlib.( - ) and ( * ) = Stdlib.( * ) in
  if k <= 0 then
    invalid_arg (Printf.sprintf "topk: k must be > 0, got %d" k);
  let ndim = List.length t.shape in
  let ax = if axis < 0 then ndim + axis else axis in
  if ax < 0 || ax >= ndim then
    invalid_arg (Printf.sprintf "topk: axis %d out of range for %d-D tensor" axis ndim);
  let axis_dim = List.nth t.shape ax in
  if k > axis_dim then
    invalid_arg (Printf.sprintf "topk: k=%d exceeds axis dimension %d" k axis_dim);
  let _t = !realize_ref t in
  let vals = Array.of_list (to_float_list _t) in
  let outer_size = ref 1 in
  for i = 0 to ax - 1 do outer_size := !outer_size * List.nth t.shape i done;
  let inner_size = ref 1 in
  for i = ax + 1 to ndim - 1 do inner_size := !inner_size * List.nth t.shape i done;
  let out_size = !outer_size * k * !inner_size in
  let top_vals = Array.make out_size 0.0 in
  let top_idxs = Array.make out_size 0.0 in
  for o = 0 to !outer_size - 1 do
    for i = 0 to !inner_size - 1 do
      let pairs = Array.init axis_dim (fun a ->
        let flat = (o * axis_dim + a) * !inner_size + i in
        (vals.(flat), a)
      ) in
      Array.sort (fun (v1, _) (v2, _) -> Float.compare v2 v1) pairs;
      for j = 0 to k - 1 do
        let (v, idx) = pairs.(j) in
        let out_flat = (o * k + j) * !inner_size + i in
        top_vals.(out_flat) <- v;
        top_idxs.(out_flat) <- Float.of_int idx
      done
    done
  done;
  let out_shape = List.mapi (fun i d -> if i = ax then k else d) t.shape in
  let v_t = from_float_list ~device:t.device ~dtype:t.dtype out_shape (Array.to_list top_vals) in
  let i_t = from_float_list ~device:t.device ~dtype:t.dtype out_shape (Array.to_list top_idxs) in
  (v_t, i_t)

(** Gather values along an axis using integer indices (host-side).
    src: N-D tensor, index: N-D tensor of integer indices (as floats).
    Output shape = index shape. For each position, selects from src along
    the given axis at the index value. *)
let gather ?(axis=0) (src : t) (index : t) : t =
  let ( + ) = Stdlib.( + ) and ( - ) = Stdlib.( - )
  and ( * ) = Stdlib.( * ) and ( / ) = Stdlib.( / ) in
  let ndim = List.length src.shape in
  let ax = if axis < 0 then ndim + axis else axis in
  if ax < 0 || ax >= ndim then
    invalid_arg (Printf.sprintf "gather: axis %d out of range for %d-D tensor" axis ndim);
  if List.length index.shape <> ndim then
    invalid_arg "gather: index must have same number of dimensions as src";
  List.iteri (fun i (sd, id) ->
    if i <> ax && sd <> id then
      invalid_arg (Printf.sprintf "gather: dimension %d mismatch: src=%d, index=%d" i sd id)
  ) (List.combine src.shape index.shape);
  let src_r = !realize_ref src in
  let idx_r = !realize_ref index in
  let src_vals = Array.of_list (to_float_list src_r) in
  let idx_vals = Array.of_list (to_float_list idx_r) in
  let out_size = Helpers.prod index.shape in
  let out_vals = Array.make out_size 0.0 in
  let src_strides = Array.make ndim 1 in
  for i = ndim - 2 downto 0 do
    src_strides.(i) <- src_strides.(i + 1) * List.nth src.shape (i + 1)
  done;
  let idx_strides = Array.make ndim 1 in
  for i = ndim - 2 downto 0 do
    idx_strides.(i) <- idx_strides.(i + 1) * List.nth index.shape (i + 1)
  done;
  for flat = 0 to out_size - 1 do
    let remaining = ref flat in
    let coords = Array.make ndim 0 in
    for d = 0 to ndim - 1 do
      coords.(d) <- !remaining / idx_strides.(d);
      remaining := !remaining - coords.(d) * idx_strides.(d)
    done;
    let idx_val = Float.to_int idx_vals.(flat) in
    let src_flat = ref 0 in
    for d = 0 to ndim - 1 do
      let c = if d = ax then idx_val else coords.(d) in
      src_flat := !src_flat + c * src_strides.(d)
    done;
    out_vals.(flat) <- src_vals.(!src_flat)
  done;
  from_float_list ~device:src.device ~dtype:src.dtype index.shape (Array.to_list out_vals)

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

(** Matrix multiply: a[N, K] @ b[K, M] = c[N, M].
    Both inputs must be exactly 2-D.
    Implemented via reshape + expand + elementwise mul + sum reduction,
    matching tinygrad's approach of decomposing matmul into primitive ops. *)
let matmul (a : t) (b : t) =
  if List.length a.shape <> 2 || List.length b.shape <> 2 then
    failwith (Printf.sprintf "matmul: both tensors must be exactly 2-D, got [%s] and [%s]"
      (String.concat "," (List.map string_of_int a.shape))
      (String.concat "," (List.map string_of_int b.shape)));
  let n = List.nth a.shape 0 in
  let k_a = List.nth a.shape 1 in
  let k_b = List.nth b.shape 0 in
  let m = List.nth b.shape 1 in
  if k_a <> k_b then
    failwith (Printf.sprintf "matmul: inner dimensions don't match (%d vs %d)" k_a k_b);
  let k = k_a in
  (* a: [N, K] → [N, K, 1] → expand [N, K, M] *)
  let a3 = reshape a [n; k; 1] in
  let a_exp = expand a3 [n; k; m] in
  (* b: [K, M] → [1, K, M] → expand [N, K, M] *)
  let b3 = reshape b [1; k; m] in
  let b_exp = expand b3 [n; k; m] in
  (* Elementwise multiply, then reduce over K (axis=1) *)
  let prod = mul a_exp b_exp in
  let summed = sum ~axes:[1] prod in  (* [N, 1, M] *)
  reshape summed [n; m]  (* [N, M] *)

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

(** Element-wise power: base ** exponent.
    Implemented as exp2(exponent * log2(abs(base))), valid for positive base. *)
let pow_ (base : t) (exponent : t) =
  let out_shape = broadcast_shape base.shape exponent.shape in
  let base' = broadcast_to base out_shape in
  let exponent' = broadcast_to exponent out_shape in
  exp (mul exponent' (log (abs_ base')))

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
  let ndim = List.length t.shape in
  let axis = if axis < 0 then ndim + axis else axis in
  let dim_size = List.nth t.shape axis in
  let base = dim_size / n in
  let remainder = dim_size mod n in
  let sizes = List.init n (fun i -> if i < remainder then base + 1 else base) in
  split ~axis t sizes

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

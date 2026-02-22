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

(** Elementwise binary operations *)
let binop op (a : t) (b : t) =
  assert (a.shape = b.shape);
  let result_dtype = match op with
    | Ops.CMPLT | Ops.CMPNE | Ops.CMPEQ -> Dtype.bool
    | _ -> a.dtype
  in
  let uop = Uop.alu op a.dtype [a.uop; b.uop] in
  { a with uop; dtype = result_dtype; lazy_uop = None }

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
  assert (cond.shape = t.shape && t.shape = f.shape);
  let uop = Uop.where_ cond.uop t.uop f.uop in
  { t with uop; lazy_uop = None }

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

(** Contiguous *)
let contiguous (t : t) =
  let uop = Uop.contiguous t.uop in
  { t with uop; lazy_uop = None }

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

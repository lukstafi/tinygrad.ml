(** Neural network building blocks, ported from tinygrad/nn/__init__.py.
    Provides Linear, BatchNorm-style layers, and SGD optimizer as
    composable modules over Tensor. *)

(** A linear (fully-connected) layer: y = x @ W + b, where W is [in_features; out_features] *)
type linear = {
  weight: Tensor.t;
  bias: Tensor.t option;
  in_features: int;
  out_features: int;
}

(** Create a Linear layer with Kaiming-uniform initialization *)
let linear ?(device="CPU") ?(dtype=Dtype.float32) ?(bias=true) ~in_features ~out_features () =
  let w = Tensor.kaiming_uniform ~device ~dtype ~fan_in:in_features [in_features; out_features] in
  let b = if bias then
    Some (Tensor.zeros ~device ~dtype [1; out_features])
  else None in
  { weight = w; bias = b; in_features; out_features }

(** Forward pass through a linear layer: x @ W + b.
    Bias is auto-broadcast from [1; out_features] to [batch; out_features]. *)
let linear_forward (layer : linear) (x : Tensor.t) : Tensor.t =
  let out = Tensor.matmul x layer.weight in
  match layer.bias with
  | Some b -> Tensor.add out b  (* auto-broadcast handles [1;out] + [batch;out] *)
  | None -> out

(** Get all trainable parameters from a linear layer *)
let linear_params (layer : linear) : Tensor.t list =
  match layer.bias with
  | Some b -> [layer.weight; b]
  | None -> [layer.weight]

(** A simple sequential model: list of (name, forward_fn, params_fn) *)
type layer = {
  name: string;
  forward: Tensor.t -> Tensor.t;
  params: unit -> Tensor.t list;
}

(** Wrap a linear layer *)
let of_linear name (l : linear) : layer =
  { name;
    forward = linear_forward l;
    params = (fun () -> linear_params l) }

(** Wrap an activation (no params) *)
let activation name (f : Tensor.t -> Tensor.t) : layer =
  { name;
    forward = f;
    params = (fun () -> []) }

(** Sequential forward: apply layers in order *)
let sequential_forward (layers : layer list) (x : Tensor.t) : Tensor.t =
  List.fold_left (fun acc l -> l.forward acc) x layers

(** Collect all parameters from a sequential model *)
let sequential_params (layers : layer list) : Tensor.t list =
  List.concat_map (fun l -> l.params ()) layers

(** Simple SGD optimizer step.
    Updates parameters in-place by creating new tensors with updated values.
    Returns (updated_params, loss_value) for the caller to replace references. *)
let sgd_step ~lr (grads : (Tensor.t * Tensor.t) list) : (Tensor.t * Tensor.t) list =
  List.map (fun (param, grad) ->
    let pv = Tensor.to_float_list param in
    let gv = Tensor.to_float_list grad in
    let new_data = List.map2 (fun p g -> p -. lr *. g) pv gv in
    let new_t = Tensor.from_float_list ~device:param.device ~dtype:param.dtype param.shape new_data in
    (param, new_t)
  ) grads

(** Batch normalization layer.
    During eval: y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
    During training (simplified): uses batch statistics. *)
type batch_norm = {
  weight: Tensor.t;         (** Scale (gamma), shape [num_features] *)
  bn_bias: Tensor.t;        (** Shift (beta), shape [num_features] *)
  running_mean: float array; (** Running mean for eval mode *)
  running_var: float array;  (** Running var for eval mode *)
  num_features: int;
  eps: float;
  momentum: float;
}

(** Create a BatchNorm layer *)
let batch_norm ?(device="CPU") ?(dtype=Dtype.float32) ?(eps=1e-5) ?(momentum=0.1) num_features =
  { weight = Tensor.ones ~device ~dtype [num_features];
    bn_bias = Tensor.zeros ~device ~dtype [num_features];
    running_mean = Array.make num_features 0.0;
    running_var = Array.make num_features 1.0;
    num_features; eps; momentum }

(** Forward pass for batch normalization (eval mode).
    Input x: [batch; channels; ...], normalizes over all dims except dim 1. *)
let batch_norm_forward (bn : batch_norm) (x : Tensor.t) : Tensor.t =
  let ndim = List.length x.shape in
  if ndim < 2 then invalid_arg "BatchNorm: input must have at least 2 dimensions";
  (* Use running statistics (eval mode) *)
  let rm = Tensor.from_float_list ~device:x.device ~dtype:x.dtype
    [bn.num_features] (Array.to_list bn.running_mean) in
  let rv = Tensor.from_float_list ~device:x.device ~dtype:x.dtype
    [bn.num_features] (Array.to_list bn.running_var) in
  (* Reshape [C] → [1, C, 1, 1, ...] for broadcasting *)
  let bc_shape = List.init ndim (fun i -> if i = 1 then bn.num_features else 1) in
  let rm_bc = Tensor.reshape rm bc_shape in
  let rv_bc = Tensor.reshape rv bc_shape in
  let w_bc = Tensor.reshape bn.weight bc_shape in
  let b_bc = Tensor.reshape bn.bn_bias bc_shape in
  let eps_t = Tensor.const_like rv_bc bn.eps in
  (* Normalize: (x - mean) / sqrt(var + eps) * weight + bias *)
  let normed = Tensor.div (Tensor.sub x rm_bc) (Tensor.sqrt_ (Tensor.add rv_bc eps_t)) in
  Tensor.add (Tensor.mul normed w_bc) b_bc

(** Get trainable parameters from a batch_norm layer *)
let batch_norm_params (bn : batch_norm) : Tensor.t list =
  [bn.weight; bn.bn_bias]

(** Wrap a batch_norm layer as a generic layer *)
let of_batch_norm name (bn : batch_norm) : layer =
  { name;
    forward = batch_norm_forward bn;
    params = (fun () -> batch_norm_params bn) }

(** Embedding layer: maps integer indices to dense vectors.
    weight shape: [num_embeddings; embedding_dim] *)
type embedding = {
  emb_weight: Tensor.t;
  num_embeddings: int;
  embedding_dim: int;
}

(** Create an embedding layer with random normal initialization *)
let embedding ?(device="CPU") ?(dtype=Dtype.float32) ~num_embeddings ~embedding_dim () =
  let w = Tensor.randn ~device ~dtype [num_embeddings; embedding_dim] in
  { emb_weight = w; num_embeddings; embedding_dim }

(** Forward pass: look up embeddings by index via one_hot @ weight.
    Input: [batch] integer tensor → Output: [batch; embedding_dim] *)
let embedding_forward (emb : embedding) (indices : Tensor.t) : Tensor.t =
  let oh = Tensor.one_hot ~num_classes:emb.num_embeddings indices in
  Tensor.matmul oh emb.emb_weight

(** Get trainable parameters from an embedding layer *)
let embedding_params (emb : embedding) : Tensor.t list = [emb.emb_weight]

(** Wrap an embedding layer as a generic layer *)
let of_embedding name (emb : embedding) : layer =
  { name;
    forward = embedding_forward emb;
    params = (fun () -> embedding_params emb) }

(** Single-head self-attention layer.
    Projects input [seq; d_model] → Q, K, V via learned linear projections,
    then applies scaled dot-product attention. *)
type self_attention = {
  wq: linear;
  wk: linear;
  wv: linear;
  wo: linear;  (** Output projection *)
  d_model: int;
}

(** Create a self-attention layer *)
let self_attention ?(device="CPU") ?(dtype=Dtype.float32) ~d_model () =
  { wq = linear ~device ~dtype ~bias:false ~in_features:d_model ~out_features:d_model ();
    wk = linear ~device ~dtype ~bias:false ~in_features:d_model ~out_features:d_model ();
    wv = linear ~device ~dtype ~bias:false ~in_features:d_model ~out_features:d_model ();
    wo = linear ~device ~dtype ~bias:false ~in_features:d_model ~out_features:d_model ();
    d_model }

(** Forward pass for self-attention.
    x: [seq; d_model], optional mask: [seq; seq].
    Returns: [seq; d_model]. *)
let self_attention_forward ?mask (attn : self_attention) (x : Tensor.t) : Tensor.t =
  let q = linear_forward attn.wq x in
  let k = linear_forward attn.wk x in
  let v = linear_forward attn.wv x in
  let attended = Tensor.scaled_dot_product_attention ?mask q k v in
  linear_forward attn.wo attended

(** Get all trainable parameters from a self-attention layer *)
let self_attention_params (attn : self_attention) : Tensor.t list =
  linear_params attn.wq @ linear_params attn.wk @
  linear_params attn.wv @ linear_params attn.wo

(** Wrap a self-attention layer *)
let of_self_attention ?mask name (attn : self_attention) : layer =
  { name;
    forward = self_attention_forward ?mask attn;
    params = (fun () -> self_attention_params attn) }

(** Adam optimizer state *)
type adam_state = {
  m: float array;   (** First moment estimate *)
  v: float array;   (** Second moment estimate *)
  t_step: int;      (** Timestep *)
}

(** Create initial Adam state for a parameter *)
let adam_init (n : int) : adam_state =
  { m = Array.make n 0.0; v = Array.make n 0.0; t_step = 0 }

(** Adam optimizer step for one parameter.
    Returns (new_param_tensor, updated_state). *)
let adam_step ?(lr=0.001) ?(beta1=0.9) ?(beta2=0.999) ?(eps=1e-8)
    (param : Tensor.t) (grad : Tensor.t) (state : adam_state) : Tensor.t * adam_state =
  let pv = Array.of_list (Tensor.to_float_list param) in
  let gv = Array.of_list (Tensor.to_float_list grad) in
  let n = Array.length pv in
  let t = state.t_step + 1 in
  let m = Array.copy state.m in
  let v = Array.copy state.v in
  for i = 0 to n - 1 do
    m.(i) <- beta1 *. m.(i) +. (1.0 -. beta1) *. gv.(i);
    v.(i) <- beta2 *. v.(i) +. (1.0 -. beta2) *. gv.(i) *. gv.(i);
  done;
  (* Bias-corrected estimates *)
  let bc1 = 1.0 -. (beta1 ** Float.of_int t) in
  let bc2 = 1.0 -. (beta2 ** Float.of_int t) in
  let new_pv = Array.init n (fun i ->
    let m_hat = m.(i) /. bc1 in
    let v_hat = v.(i) /. bc2 in
    pv.(i) -. lr *. m_hat /. (Stdlib.sqrt v_hat +. eps)
  ) in
  let new_t = Tensor.from_float_list ~device:param.device ~dtype:param.dtype param.shape
    (Array.to_list new_pv) in
  (new_t, { m; v; t_step = t })

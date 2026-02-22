(** Neural network building blocks, ported from tinygrad/nn/__init__.py.
    Provides Linear, BatchNorm-style layers, and SGD optimizer as
    composable modules over Tensor. *)

(** A linear (fully-connected) layer: y = x @ W^T + b *)
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

(** Forward pass through a linear layer: x @ W + b *)
let linear_forward (layer : linear) (x : Tensor.t) : Tensor.t =
  let batch = List.hd x.shape in
  let out = Tensor.matmul x layer.weight in
  match layer.bias with
  | Some b -> Tensor.add out (Tensor.expand b [batch; layer.out_features])
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

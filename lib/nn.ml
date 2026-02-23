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

(** Batch normalization layer (eval mode only).
    y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
    Note: this implementation only supports inference/eval mode.
    Running statistics must be pre-populated; they are not updated during forward. *)
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

(** Forward pass for batch normalization (eval mode only).
    Input x: [batch; channels; ...], normalizes over all dims except dim 1.
    Uses pre-populated running_mean and running_var; does not compute batch statistics. *)
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

(** Layer normalization.
    Normalizes over the last [normalized_shape] dimensions.
    y = (x - mean) / sqrt(var + eps) * weight + bias *)
type layer_norm = {
  ln_weight: Tensor.t;       (** Scale (gamma) *)
  ln_bias: Tensor.t;         (** Shift (beta) *)
  normalized_shape: int list; (** Shape of the dimensions being normalized *)
  ln_eps: float;
}

(** Create a LayerNorm layer *)
let layer_norm ?(device="CPU") ?(dtype=Dtype.float32) ?(eps=1e-5) (normalized_shape : int list) =
  let size = List.fold_left ( * ) 1 normalized_shape in
  { ln_weight = Tensor.from_float_list ~device ~dtype [size] (List.init size (fun _ -> 1.0));
    ln_bias = Tensor.from_float_list ~device ~dtype [size] (List.init size (fun _ -> 0.0));
    normalized_shape;
    ln_eps = eps }

(** Forward pass for layer normalization.
    Normalizes over the last N dimensions matching normalized_shape.
    Input x: [...; *normalized_shape], output same shape. *)
let layer_norm_forward (ln : layer_norm) (x : Tensor.t) : Tensor.t =
  let ndim = List.length x.shape in
  let norm_ndim = List.length ln.normalized_shape in
  if ndim < norm_ndim then
    invalid_arg "LayerNorm: input has fewer dims than normalized_shape";
  let trailing = List.filteri (fun i _ -> i >= ndim - norm_ndim) x.shape in
  if trailing <> ln.normalized_shape then
    invalid_arg (Printf.sprintf "LayerNorm: trailing dims %s != normalized_shape %s"
      (String.concat "," (List.map string_of_int trailing))
      (String.concat "," (List.map string_of_int ln.normalized_shape)));
  (* Reduce over the last norm_ndim dimensions *)
  let reduce_axes = List.init norm_ndim (fun i -> ndim - norm_ndim + i) in
  let m = Tensor.mean ~axes:reduce_axes x in
  let m_exp = Tensor.expand m x.shape in
  let diff = Tensor.sub x m_exp in
  let var = Tensor.mean ~axes:reduce_axes (Tensor.mul diff diff) in
  let var_exp = Tensor.expand var x.shape in
  let eps_t = Tensor.const_like var_exp ln.ln_eps in
  let normed = Tensor.div diff (Tensor.sqrt_ (Tensor.add var_exp eps_t)) in
  (* Reshape weight/bias to broadcast: [1, ..., 1, *normalized_shape] *)
  let bc_shape = List.init ndim (fun i ->
    if i < ndim - norm_ndim then 1
    else List.nth ln.normalized_shape (i - (ndim - norm_ndim))
  ) in
  let w = Tensor.expand (Tensor.reshape ln.ln_weight bc_shape) x.shape in
  let b = Tensor.expand (Tensor.reshape ln.ln_bias bc_shape) x.shape in
  Tensor.add (Tensor.mul normed w) b

(** Get trainable parameters from a layer_norm layer *)
let layer_norm_params (ln : layer_norm) : Tensor.t list =
  [ln.ln_weight; ln.ln_bias]

(** Wrap a layer_norm layer as a generic layer *)
let of_layer_norm name (ln : layer_norm) : layer =
  { name;
    forward = layer_norm_forward ln;
    params = (fun () -> layer_norm_params ln) }

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
    Input: [batch] integer tensor → Output: [batch; embedding_dim].
    Validates that all indices are in [0, num_embeddings).
    Out-of-range indices would silently produce zero rows in the one_hot encoding. *)
let embedding_forward (emb : embedding) (indices : Tensor.t) : Tensor.t =
  (* Validate indices: must be integers in range [0, num_embeddings) *)
  let idx_vals = Tensor.to_float_list indices in
  List.iteri (fun i v ->
    if Float.of_int (Float.to_int v) <> v then
      invalid_arg (Printf.sprintf
        "Embedding: non-integer index %g at position %d" v i);
    let idx = Float.to_int v in
    if idx < 0 || idx >= emb.num_embeddings then
      invalid_arg (Printf.sprintf
        "Embedding: index %d at position %d out of range [0, %d)"
        idx i emb.num_embeddings)
  ) idx_vals;
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

(** Flatten layer wrapper — reshapes to [batch; -1] or to flat vector.
    start_dim: first dimension to flatten (default 1, preserving batch dim).
    For 1D input (no batch), start_dim=0 will flatten everything. *)
let flatten_layer ?(start_dim=1) name : layer =
  { name;
    forward = (fun (x : Tensor.t) ->
      let shape = x.shape in
      let ndim = List.length shape in
      let sd = if start_dim < 0 then ndim + start_dim else start_dim in
      if sd < 0 || sd > ndim then
        invalid_arg (Printf.sprintf "flatten_layer: start_dim=%d out of range for %d-D input" start_dim ndim);
      let prefix = List.filteri (fun i _ -> i < sd) shape in
      let suffix = List.filteri (fun i _ -> i >= sd) shape in
      let flat_dim = List.fold_left ( * ) 1 suffix in
      Tensor.reshape x (prefix @ [flat_dim])
    );
    params = (fun () -> []) }

(** Dropout layer wrapper — applies dropout during training.
    p: dropout probability (default 0.5).
    training: if false, acts as identity (default true). *)
let dropout_layer ?(p=0.5) ?(training=true) name : layer =
  { name;
    forward = (fun (x : Tensor.t) ->
      if training then Tensor.dropout ~p x else x
    );
    params = (fun () -> []) }

(** Multi-head attention layer.
    Splits d_model into n_heads, applies attention per head, concatenates.
    x: [seq; d_model], returns: [seq; d_model]. *)
type multi_head_attention = {
  mha_wq: linear;
  mha_wk: linear;
  mha_wv: linear;
  mha_wo: linear;
  mha_d_model: int;
  mha_n_heads: int;
  mha_head_dim: int;
}

let multi_head_attention ?(device="CPU") ?(dtype=Dtype.float32) ~d_model ~n_heads () =
  if n_heads <= 0 then
    invalid_arg (Printf.sprintf "multi_head_attention: n_heads must be > 0, got %d" n_heads);
  if d_model mod n_heads <> 0 then
    invalid_arg (Printf.sprintf "multi_head_attention: d_model=%d not divisible by n_heads=%d" d_model n_heads);
  let head_dim = d_model / n_heads in
  { mha_wq = linear ~device ~dtype ~bias:false ~in_features:d_model ~out_features:d_model ();
    mha_wk = linear ~device ~dtype ~bias:false ~in_features:d_model ~out_features:d_model ();
    mha_wv = linear ~device ~dtype ~bias:false ~in_features:d_model ~out_features:d_model ();
    mha_wo = linear ~device ~dtype ~bias:false ~in_features:d_model ~out_features:d_model ();
    mha_d_model = d_model;
    mha_n_heads = n_heads;
    mha_head_dim = head_dim }

(** Multi-head attention forward pass.
    Processes each head separately (seq-only, no batch dim for now).
    x: [seq; d_model], optional mask: [seq; seq]. Returns: [seq; d_model]. *)
let multi_head_attention_forward ?mask (mha : multi_head_attention) (x : Tensor.t) : Tensor.t =
  let seq_len = List.hd x.shape in
  let q_full = linear_forward mha.mha_wq x in
  let k_full = linear_forward mha.mha_wk x in
  let v_full = linear_forward mha.mha_wv x in
  (* Process each head via slicing: head i uses dims [i*head_dim .. (i+1)*head_dim) *)
  let head_outputs = List.init mha.mha_n_heads (fun h ->
    let start = h * mha.mha_head_dim in
    let stop = start + mha.mha_head_dim in
    (* Slice: shrink along last dimension *)
    let q_h = Tensor.shrink q_full [(0, seq_len); (start, stop)] in
    let k_h = Tensor.shrink k_full [(0, seq_len); (start, stop)] in
    let v_h = Tensor.shrink v_full [(0, seq_len); (start, stop)] in
    (* Realize intermediates to keep graph manageable *)
    let q_h = !(Tensor.realize_ref) q_h in
    let k_h = !(Tensor.realize_ref) k_h in
    let v_h = !(Tensor.realize_ref) v_h in
    (* Apply scaled dot-product attention for this head *)
    Tensor.scaled_dot_product_attention ?mask q_h k_h v_h
  ) in
  (* Concatenate heads along last dimension *)
  let concat = Tensor.cat ~axis:1 head_outputs in
  let concat = !(Tensor.realize_ref) concat in
  linear_forward mha.mha_wo concat

let multi_head_attention_params (mha : multi_head_attention) : Tensor.t list =
  linear_params mha.mha_wq @ linear_params mha.mha_wk @
  linear_params mha.mha_wv @ linear_params mha.mha_wo

let of_multi_head_attention ?mask name (mha : multi_head_attention) : layer =
  { name;
    forward = multi_head_attention_forward ?mask mha;
    params = (fun () -> multi_head_attention_params mha) }

(** 2D convolution layer (forward/inference only — gradients do not flow through weights).
    weight: [out_channels, in_channels, kernel_h, kernel_w]
    bias: [out_channels] (optional) *)
type conv2d = {
  conv_weight: Tensor.t;
  conv_bias: Tensor.t option;
  conv_stride: int;
  conv_padding: int;
  out_channels: int;
  in_channels: int;
  kernel_size: int * int;
}

let conv2d ?(device="CPU") ?(dtype=Dtype.float32) ?(bias=true)
    ?(stride=1) ?(padding=0) ~in_channels ~out_channels ~kernel_size () =
  let kh, kw = kernel_size in
  let fan_in = in_channels * kh * kw in
  let w = Tensor.kaiming_uniform ~device ~dtype [out_channels; in_channels; kh; kw] ~fan_in in
  let b = if bias then
    Some (Tensor.zeros ~device ~dtype [out_channels])
  else None in
  { conv_weight = w; conv_bias = b; conv_stride = stride;
    conv_padding = padding; out_channels; in_channels;
    kernel_size = (kh, kw) }

let conv2d_forward (c : conv2d) (x : Tensor.t) : Tensor.t =
  let out = Tensor.conv2d ~stride:c.conv_stride ~padding:c.conv_padding x c.conv_weight in
  match c.conv_bias with
  | None -> out
  | Some b ->
    (* bias: [out_channels] → [out_channels, 1, 1] → broadcast *)
    let oh = List.nth out.shape 1 in
    let ow = List.nth out.shape 2 in
    let b3 = Tensor.reshape b [c.out_channels; 1; 1] in
    let b_exp = Tensor.expand b3 [c.out_channels; oh; ow] in
    Tensor.add out b_exp

let conv2d_params (c : conv2d) : Tensor.t list =
  match c.conv_bias with
  | None -> [c.conv_weight]
  | Some b -> [c.conv_weight; b]

let of_conv2d name (c : conv2d) : layer =
  { name;
    forward = conv2d_forward c;
    params = (fun () -> conv2d_params c) }

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

(** AdamW optimizer step: Adam with decoupled weight decay.
    weight_decay is applied directly to the parameter (not through gradient). *)
let adamw_step ?(lr=0.001) ?(beta1=0.9) ?(beta2=0.999) ?(eps=1e-8) ?(weight_decay=0.01)
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
  let bc1 = 1.0 -. (beta1 ** Float.of_int t) in
  let bc2 = 1.0 -. (beta2 ** Float.of_int t) in
  let new_pv = Array.init n (fun i ->
    let m_hat = m.(i) /. bc1 in
    let v_hat = v.(i) /. bc2 in
    (* Decoupled weight decay: applied to parameter directly *)
    let p_decayed = pv.(i) *. (1.0 -. lr *. weight_decay) in
    p_decayed -. lr *. m_hat /. (Stdlib.sqrt v_hat +. eps)
  ) in
  let new_t = Tensor.from_float_list ~device:param.device ~dtype:param.dtype param.shape
    (Array.to_list new_pv) in
  (new_t, { m; v; t_step = t })

(* ---- Gradient Clipping ---- *)

(** Clip gradient values element-wise to [-clip_value, clip_value].
    Returns a new gradient tensor list with clipped values. *)
let clip_grad_value ~clip_value (grads : (Tensor.t * Tensor.t) list)
    : (Tensor.t * Tensor.t) list =
  List.map (fun (param, (grad : Tensor.t)) ->
    let gv = Tensor.to_float_list grad in
    let clipped = List.map (fun g ->
      Float.max (-.clip_value) (Float.min clip_value g)
    ) gv in
    let new_g = Tensor.from_float_list ~device:grad.device ~dtype:grad.dtype
      grad.shape clipped in
    (param, new_g)
  ) grads

(** Clip gradients by global L2 norm.
    If total_norm > max_norm, scale all gradients by max_norm / total_norm.
    Returns (clipped_grads, total_norm). *)
let clip_grad_norm ~max_norm (grads : (Tensor.t * Tensor.t) list)
    : (Tensor.t * Tensor.t) list * float =
  let total_sq = List.fold_left (fun acc (_, grad) ->
    let gv = Tensor.to_float_list grad in
    List.fold_left (fun a g -> a +. g *. g) acc gv
  ) 0.0 grads in
  let total_norm = Stdlib.sqrt total_sq in
  if total_norm <= max_norm then (grads, total_norm)
  else
    let scale = max_norm /. total_norm in
    let clipped = List.map (fun (param, (grad : Tensor.t)) ->
      let gv = Tensor.to_float_list grad in
      let scaled = List.map (fun g -> g *. scale) gv in
      let new_g = Tensor.from_float_list ~device:grad.device ~dtype:grad.dtype
        grad.shape scaled in
      (param, new_g)
    ) grads in
    (clipped, total_norm)

(* ---- Learning Rate Schedulers ---- *)

(** Learning rate scheduler state *)
type lr_scheduler = {
  base_lr: float;
  current_lr: float;
  step_count: int;
}

(** Create a new LR scheduler with the given base learning rate *)
let lr_scheduler_init (base_lr : float) : lr_scheduler =
  { base_lr; current_lr = base_lr; step_count = 0 }

(** Step decay: multiply LR by gamma every step_size steps.
    lr = base_lr * gamma^(step_count / step_size) *)
let lr_step_decay ~step_size ~gamma (sched : lr_scheduler) : lr_scheduler =
  if step_size <= 0 then invalid_arg "lr_step_decay: step_size must be > 0";
  let step = sched.step_count + 1 in
  let lr = sched.base_lr *. (gamma ** Float.of_int (step / step_size)) in
  { base_lr = sched.base_lr; current_lr = lr; step_count = step }

(** Exponential decay: multiply LR by gamma each step.
    lr = base_lr * gamma^step_count *)
let lr_exponential_decay ~gamma (sched : lr_scheduler) : lr_scheduler =
  let step = sched.step_count + 1 in
  let lr = sched.base_lr *. (gamma ** Float.of_int step) in
  { base_lr = sched.base_lr; current_lr = lr; step_count = step }

(** Cosine annealing: lr oscillates between base_lr and eta_min over T_max steps.
    lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * step / T_max)) *)
let lr_cosine_annealing ~t_max ?(eta_min=0.0) (sched : lr_scheduler) : lr_scheduler =
  if t_max <= 0 then invalid_arg "lr_cosine_annealing: t_max must be > 0";
  let step = sched.step_count + 1 in
  let progress = Float.of_int step /. Float.of_int t_max in
  let lr = eta_min +. 0.5 *. (sched.base_lr -. eta_min) *.
    (1.0 +. Stdlib.cos (Float.pi *. progress)) in
  { base_lr = sched.base_lr; current_lr = lr; step_count = step }

(* ---- Model Save/Load ---- *)

(** Save model parameters to a file.
    Format: one line per parameter with "name shape data..." *)
let save_params (filename : string) (params : (string * Tensor.t) list) : unit =
  let oc = open_out filename in
  List.iter (fun (name, (tensor : Tensor.t)) ->
    let shape_str = String.concat "," (List.map string_of_int tensor.shape) in
    let values = Tensor.to_float_list tensor in
    let val_str = String.concat " " (List.map (Printf.sprintf "%.17g") values) in
    Printf.fprintf oc "%s|%s|%s\n" name shape_str val_str
  ) params;
  close_out oc

(** Load model parameters from a file.
    Returns a list of (name, float_array, shape) for the caller to apply. *)
let load_params ?(device="CPU") ?(dtype=Dtype.float32) (filename : string)
    : (string * Tensor.t) list =
  let ic = open_in filename in
  let params = ref [] in
  (try while true do
    let line = input_line ic in
    match String.split_on_char '|' line with
    | [name; shape_str; val_str] ->
      let shape = if shape_str = "" then []
        else List.map int_of_string (String.split_on_char ',' shape_str) in
      let values = List.filter_map (fun s ->
        if s = "" then None else Some (float_of_string s)
      ) (String.split_on_char ' ' val_str) in
      let tensor = Tensor.from_float_list ~device ~dtype shape values in
      params := (name, tensor) :: !params
    | _ -> ()
  done with End_of_file -> ());
  close_in ic;
  List.rev !params

(** Save sequential model parameters with layer names *)
let save_sequential (filename : string) (layers : layer list) : unit =
  let params = List.concat_map (fun l ->
    List.mapi (fun i t -> (Printf.sprintf "%s.%d" l.name i, t)) (l.params ())
  ) layers in
  save_params filename params

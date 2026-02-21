(** Autograd via pattern-matched differentiation.
    Ported from tinygrad/gradient.py.

    Each differentiable op has a gradient rule expressed as a pattern match.
    compute_gradient walks the graph in reverse toposort, accumulating gradients. *)

(** Compute the gradient of one op given its output gradient (ctx) and the op's UOp (ret).
    Returns a list of gradients for each source, or None for non-differentiable sources. *)
let grad_for_op (ctx : Uop.t) (ret : Uop.t) : Uop.t option list =
  match ret.op with
  | Ops.CAST ->
    let src = List.hd ret.src in
    [Some (Uop.cast src.dtype ctx)]
  | Ops.RECIPROCAL ->
    (* d/dx (1/x) = -1/x^2 = -ret^2 * ctx *)
    [Some (Uop.neg (Uop.mul (Uop.mul ret ret) ctx))]
  | Ops.SIN ->
    (* d/dx sin(x) = cos(x) * ctx = sin(pi/2 - x) * ctx *)
    let x = List.hd ret.src in
    let half_pi = Uop.const_float x.dtype (Float.pi /. 2.0) in
    let cos_x = Uop.sin_ (Uop.sub half_pi x) in
    [Some (Uop.mul cos_x ctx)]
  | Ops.LOG2 ->
    (* d/dx log2(x) = 1/(x * ln(2)) * ctx *)
    let x = List.hd ret.src in
    let ln2 = Uop.const_float x.dtype (Float.log 2.0) in
    [Some (Uop.mul ctx (Uop.recip (Uop.mul x ln2)))]
  | Ops.EXP2 ->
    (* d/dx 2^x = 2^x * ln(2) * ctx = ret * ln(2) * ctx *)
    let ln2 = Uop.const_float ret.dtype (Float.log 2.0) in
    [Some (Uop.mul (Uop.mul ret ctx) ln2)]
  | Ops.SQRT ->
    (* d/dx sqrt(x) = 1/(2*sqrt(x)) * ctx = ctx / (2*ret) *)
    let two = Uop.const_float ret.dtype 2.0 in
    [Some (Uop.mul ctx (Uop.recip (Uop.mul two ret)))]
  | Ops.ADD ->
    [Some ctx; Some ctx]
  | Ops.SUB ->
    [Some ctx; Some (Uop.neg ctx)]
  | Ops.MUL ->
    let a = List.nth ret.src 0 in
    let b = List.nth ret.src 1 in
    [Some (Uop.mul b ctx); Some (Uop.mul a ctx)]
  | Ops.WHERE ->
    let cond = List.nth ret.src 0 in
    let zero = Uop.const ctx.dtype 0.0 in
    [None; Some (Uop.where_ cond ctx zero); Some (Uop.where_ cond zero ctx)]
  | Ops.MAX ->
    let a = List.nth ret.src 0 in
    let b = List.nth ret.src 1 in
    let a_gt_b = Uop.cmplt b a in  (* a > b *)
    let a_eq_b = Uop.cmpeq a b in
    let zero = Uop.const ctx.dtype 0.0 in
    let half = Uop.const_float ctx.dtype 0.5 in
    let half_ctx = Uop.mul half ctx in
    [Some (Uop.where_ a_gt_b ctx (Uop.where_ a_eq_b half_ctx zero));
     Some (Uop.where_ (Uop.cmplt a b) ctx (Uop.where_ a_eq_b half_ctx zero))]
  | Ops.CMPLT | Ops.CMPNE | Ops.CMPEQ ->
    [None; None]
  | Ops.NEG ->
    [Some (Uop.neg ctx)]
  | Ops.CONTIGUOUS ->
    [Some ctx]
  | Ops.RESHAPE ->
    (* gradient of reshape is reshape back to source shape *)
    let src_shape = match ret.arg with
      | Uop.Shape _ ->
        (* ret.arg has the output shape; we need the source shape.
           The source UOp carries its own shape in its arg if it's a RESHAPE,
           or we infer from EXPAND/original. For the general case, look at
           the source node's shape arg. *)
        (match (List.hd ret.src).arg with
         | Uop.Shape s -> s
         | _ -> [])  (* fallback: src has no shape info *)
      | _ -> []
    in
    if src_shape <> [] then
      [Some (Uop.reshape ctx src_shape)]
    else
      (* Can't determine source shape — pass gradient through unchanged *)
      [Some ctx]
  | Ops.EXPAND ->
    (* gradient of expand is reduce_sum over expanded dims.
       Expanded dims are those where src had size 1 but output has size > 1. *)
    let out_shape = match ret.arg with Uop.Shape s -> s | _ -> [] in
    let src = List.hd ret.src in
    let src_shape = match src.arg with Uop.Shape s -> s | _ -> [] in
    if src_shape <> [] && out_shape <> [] then begin
      (* Find dims that were expanded: src[i]=1 but out[i]>1 *)
      let expanded_axes = List.filter_map (fun i ->
        if List.nth src_shape i = 1 && List.nth out_shape i > 1 then Some i
        else None
      ) (List.init (List.length src_shape) Fun.id) in
      if expanded_axes = [] then [Some ctx]
      else
        (* Reduce sum over expanded dims, then reshape back to src_shape *)
        let reduced = Uop.reduce_axis ~src_shape:out_shape ctx Ops.ADD expanded_axes in
        [Some (Uop.reshape reduced src_shape)]
    end else
      [Some ctx]  (* fallback: pass through *)
  | Ops.PERMUTE ->
    let inv_axes = match ret.arg with
      | Uop.Int_list axes -> Helpers.argsort axes
      | _ -> []
    in
    [Some (Uop.permute ctx inv_axes)]
  | Ops.REDUCE_AXIS ->
    let reduce_op, reduce_axes, src_shape = match ret.arg with
      | Uop.Axis_arg (axes, op, src_shape) -> (op, axes, src_shape)
      | _ -> (Ops.ADD, [], [])
    in
    begin match reduce_op with
    | Ops.ADD ->
      (* d/dx sum(x, axes) = expand gradient back to src_shape.
         ctx has shape with 1s at reduced positions; expand to src_shape. *)
      if src_shape <> [] then
        [Some (Uop.expand (Uop.reshape ctx (List.mapi (fun i d ->
          if List.mem i reduce_axes then 1 else d) src_shape)) src_shape)]
      else
        [Some ctx]  (* fallback: no shape info *)
    | Ops.MAX ->
      (* d/dx max(x, axes) = gradient flows only to argmax positions.
         Simplified: use indicator mask (x == max_expanded) * ctx_expanded. *)
      if src_shape <> [] then begin
        let out_shape = List.mapi (fun i d ->
          if List.mem i reduce_axes then 1 else d) src_shape in
        let ctx_expanded = Uop.expand (Uop.reshape ctx out_shape) src_shape in
        let ret_expanded = Uop.expand (Uop.reshape ret out_shape) src_shape in
        let x = List.hd ret.src in
        let mask = Uop.cmpeq x ret_expanded in
        let zero = Uop.const ctx.dtype 0.0 in
        [Some (Uop.where_ mask ctx_expanded zero)]
      end else
        [Some ctx]
    | _ -> [Some ctx]
    end
  | _ ->
    (* No gradient defined *)
    List.map (fun _ -> None) ret.src

(** Compute gradients via reverse-mode autodiff.
    Walks the graph from root backward, accumulating gradients. *)
let compute_gradient (root : Uop.t) (root_grad : Uop.t) (targets : Uop.t list) : (Uop.t * Uop.t) list =
  let target_ids = List.map (fun u -> u.Uop.id) targets in
  let grads : (int, Uop.t) Hashtbl.t = Hashtbl.create 64 in
  Hashtbl.replace grads root.id root_grad;

  (* Check if a node is on the path to any target *)
  let on_path = Hashtbl.create 256 in
  let all = Uop.toposort1 root in
  (* Mark all nodes that lead to targets *)
  List.iter (fun u ->
    if List.mem u.Uop.id target_ids then
      Hashtbl.replace on_path u.id true
  ) all;
  (* Propagate: a node is on path if any of its sources are *)
  List.iter (fun (u : Uop.t) ->
    if List.exists (fun (s : Uop.t) -> Hashtbl.mem on_path s.id) u.src then
      Hashtbl.replace on_path u.id true
  ) all;

  (* Walk in reverse topological order *)
  let rev_all = List.rev all in
  List.iter (fun (u : Uop.t) ->
    match Hashtbl.find_opt grads u.id with
    | None -> ()
    | Some grad ->
      if Hashtbl.mem on_path u.id then begin
        let src_grads = grad_for_op grad u in
        List.iter2 (fun (src : Uop.t) grad_opt ->
          match grad_opt with
          | None -> ()
          | Some g ->
            let existing = Hashtbl.find_opt grads src.id in
            let new_grad = match existing with
              | None -> g
              | Some prev -> Uop.add prev g
            in
            Hashtbl.replace grads src.id new_grad
        ) u.src src_grads
      end
  ) rev_all;

  (* Return gradients for targets *)
  List.filter_map (fun (target : Uop.t) ->
    match Hashtbl.find_opt grads target.id with
    | Some g -> Some (target, g)
    | None -> None
  ) targets

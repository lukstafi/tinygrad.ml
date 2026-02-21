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
    let src = List.hd ret.src in
    (* Need to figure out src shape from the arg *)
    let src_shape = match src.arg with
      | Uop.Shape s -> s
      | _ -> []  (* fallback *)
    in
    ignore src_shape;
    [Some (Uop.reshape ctx []);  (* placeholder — needs src shape *) None]
  | Ops.EXPAND ->
    (* gradient of expand is reduce_sum over expanded dims *)
    [Some ctx; None]  (* simplified — full version reduces expanded dims *)
  | Ops.PERMUTE ->
    let inv_axes = match ret.arg with
      | Uop.Int_list axes -> Helpers.argsort axes
      | _ -> []
    in
    [Some (Uop.permute ctx inv_axes)]
  | Ops.REDUCE_AXIS ->
    let reduce_op = match ret.arg with Uop.Axis_arg (_, op, _) -> op | _ -> Ops.ADD in
    begin match reduce_op with
    | Ops.ADD -> [Some ctx]  (* broadcast back *)
    | _ -> [Some ctx]  (* simplified *)
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

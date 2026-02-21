(** Pattern matching and graph rewriting engine for UOp graphs.
    Ported from tinygrad/uop/ops.py (UPat, PatternMatcher, graph_rewrite).

    This is the backbone of tinygrad's compilation pipeline. Every transformation
    — from scheduling to optimization to linearization — is expressed as pattern-matched
    rewrites over the UOp graph. *)

(** A pattern for matching UOp nodes *)
type upat = {
  ops: Ops.t list option;           (** None = match any op, Some ops = match any of these ops *)
  dtype: Dtype.t option;             (** None = match any dtype, Some dt = must match exactly *)
  src: upat list option;             (** None = match any sources, Some pats = match sources against patterns *)
  name: string option;               (** If set, capture matched UOp under this name *)
  arg_match: Uop.arg option;         (** If set, arg must equal this *)
  allow_any_len: bool;               (** If true, src patterns are a prefix; extra sources are ok *)
}

(** Convenience constructors for patterns *)
let pat ?ops ?dtype ?src ?name ?arg ?(allow_any_len=false) () =
  { ops; dtype; src; name; arg_match = arg; allow_any_len }

let op_pat ops = pat ~ops ()
let var name = pat ~name ()
let any_pat = pat ()

(** A rewrite rule: pattern + replacement function.
    The replacement function receives the context (if any) and captured bindings,
    and returns Some new_uop or None (meaning no match). *)
type 'ctx rule = {
  pattern: upat;
  rewrite: 'ctx -> (string * Uop.t) list -> Uop.t option;
}

(** A pattern matcher: an ordered list of rules *)
type 'ctx t = 'ctx rule list

(** Create a pattern matcher from a list of (pattern, rewrite_fn) pairs *)
let create rules = rules

(** Concatenate two pattern matchers *)
let concat a b = a @ b

(** Match a pattern against a UOp, returning captured bindings if successful *)
let rec match_pat (pat : upat) (uop : Uop.t) : (string * Uop.t) list option =
  (* Check op *)
  let op_ok = match pat.ops with
    | None -> true
    | Some ops -> List.mem uop.op ops
  in
  if not op_ok then None
  else
  (* Check dtype *)
  let dtype_ok = match pat.dtype with
    | None -> true
    | Some dt -> uop.dtype = dt
  in
  if not dtype_ok then None
  else
  (* Check arg *)
  let arg_ok = match pat.arg_match with
    | None -> true
    | Some a -> uop.arg = a
  in
  if not arg_ok then None
  else
  (* Check sources *)
  let src_bindings = match pat.src with
    | None -> Some []  (* match any sources *)
    | Some pats ->
      let n_pats = List.length pats in
      let n_srcs = List.length uop.src in
      if pat.allow_any_len then begin
        if n_srcs < n_pats then None
        else match_sources pats (List.filteri (fun i _ -> i < n_pats) uop.src)
      end else begin
        if n_srcs <> n_pats then None
        else match_sources pats uop.src
      end
  in
  match src_bindings with
  | None -> None
  | Some bindings ->
    (* Add self to bindings if named *)
    let bindings = match pat.name with
      | Some n -> (n, uop) :: bindings
      | None -> bindings
    in
    Some bindings

and match_sources pats srcs =
  match pats, srcs with
  | [], [] -> Some []
  | [], _ | _, [] -> None
  | p :: ps, s :: ss ->
    match match_pat p s with
    | None -> None
    | Some bindings ->
      match match_sources ps ss with
      | None -> None
      | Some more_bindings -> Some (bindings @ more_bindings)

(** Try to apply a single rule to a UOp *)
let try_rule (ctx : 'ctx) (rule : 'ctx rule) (uop : Uop.t) : Uop.t option =
  match match_pat rule.pattern uop with
  | None -> None
  | Some bindings -> rule.rewrite ctx bindings

(** Try to rewrite a UOp using the first matching rule *)
let rewrite_one (pm : 'ctx t) (ctx : 'ctx) (uop : Uop.t) : Uop.t option =
  let rec try_rules = function
    | [] -> None
    | rule :: rest ->
      match try_rule ctx rule uop with
      | Some result -> Some result
      | None -> try_rules rest
  in
  try_rules pm

(** Graph rewrite: apply pattern matcher to all nodes in a UOp graph until fixed point.

    This is the core transformation engine. It:
    1. Topologically sorts the graph
    2. For each node (bottom-up by default), tries to apply the first matching rule
    3. If a rule matches, replaces the node and re-processes affected nodes
    4. Repeats until no more rewrites are possible

    Ported from tinygrad's graph_rewrite function. *)
let graph_rewrite ?(bottom_up=true) (pm : 'ctx t) (ctx : 'ctx) (root : Uop.t) : Uop.t =
  (* Single-pass bottom-up approach: process sources first, then try rewrite.
     Rewrite results are re-processed to handle cascading rewrites.
     Cache prevents reprocessing the same node. *)
  let cache : (int, Uop.t) Hashtbl.t = Hashtbl.create 256 in

  let rec process (u : Uop.t) : Uop.t =
    match Hashtbl.find_opt cache u.id with
    | Some result -> result
    | None ->
      (* Mark as being processed (prevents infinite recursion on cycles) *)
      Hashtbl.replace cache u.id u;
      (* First, recursively process all sources *)
      let new_src = if bottom_up then List.map process u.src else u.src in
      (* Rebuild with processed sources *)
      let rebuilt =
        if List.for_all2 Uop.equal u.src new_src then u
        else Uop.create ~arg:u.arg u.op u.dtype new_src
      in
      (* Try to apply rewrite rules repeatedly until no more match *)
      let rec apply_rewrites node depth =
        if depth > 100 then node  (* safety limit *)
        else match rewrite_one pm ctx node with
          | Some new_u when not (Uop.equal new_u node) ->
            (* Process the new node's sources (they might need rewriting too) *)
            let new_src2 = if bottom_up then List.map process new_u.src else new_u.src in
            let rebuilt2 =
              if List.for_all2 Uop.equal new_u.src new_src2 then new_u
              else Uop.create ~arg:new_u.arg new_u.op new_u.dtype new_src2
            in
            apply_rewrites rebuilt2 (depth + 1)
          | _ -> node
      in
      let result = apply_rewrites rebuilt 0 in
      Hashtbl.replace cache u.id result;
      result
  in
  process root

(** Helper: find a binding by name in a binding list *)
let find name bindings =
  match List.assoc_opt name bindings with
  | Some u -> u
  | None -> failwith (Printf.sprintf "pattern binding %S not found" name)

(** Helper: find an optional binding *)
let find_opt name bindings = List.assoc_opt name bindings

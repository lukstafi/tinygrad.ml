(** UOp: the core intermediate representation for tinygrad_ml.
    Ported from tinygrad/uop/ops.py (UOp class).

    Every node in the computation graph is a UOp. UOps are hash-consed: creating
    two UOps with the same (op, dtype, src, arg) returns the same physical object.
    This is critical for graph rewriting correctness (identity-based matching). *)

(** The argument type — a dynamically-typed value attached to UOp nodes.
    In Python tinygrad this is `Any`; in OCaml we use a tagged variant. *)
type arg =
  | No_arg
  | Int_arg of int
  | Float_arg of float
  | String_arg of string
  | Int_list of int list
  | Pair_int of int * int
  | Tuple_int of int list  (** generic int tuple *)
  | Shape of int list      (** shape argument for movement ops *)
  | Pad_arg of (int * int) list  (** padding: list of (before, after) *)
  | Axis_arg of int list * Ops.t * int list  (** reduce axis: (axes, reduce_op, src_shape) *)
  | Func_name of string    (** function name for SINK *)

(** A UOp node. Fields are mutable only for the hash-consing cache;
    logically they are immutable after creation. *)
type t = {
  op: Ops.t;
  dtype: Dtype.t;
  src: t list;
  arg: arg;
  id: int;  (** unique id for ordering and hashing *)
}

(** Global counter for unique IDs *)
let next_id = ref 0

(** Hash-consing cache. We use a regular Hashtbl since OCaml doesn't have
    Python's weak-value dict semantics that work well here. For a port,
    this is simpler and correct — the cache grows but UOps are typically
    not GC'd during a computation anyway. *)
let cache : (Ops.t * int * int list * int, t) Hashtbl.t = Hashtbl.create 4096

(** Reset the UOp ID counter and hash-consing cache.
    Call between independent computations (e.g. training steps) to prevent
    unbounded ID growth that causes renderer lookup failures. *)
let reset () =
  next_id := 0;
  Hashtbl.clear cache

(** Hash an arg deterministically *)
let hash_arg = function
  | No_arg -> 0
  | Int_arg i -> Hashtbl.hash i
  | Float_arg f -> Hashtbl.hash f
  | String_arg s -> Hashtbl.hash s
  | Int_list l -> Hashtbl.hash l
  | Pair_int (a, b) -> Hashtbl.hash (a, b)
  | Tuple_int l -> Hashtbl.hash l
  | Shape l -> Hashtbl.hash ("shape", l)
  | Pad_arg l -> Hashtbl.hash ("pad", l)
  | Axis_arg (axes, op, _src_shape) -> Hashtbl.hash ("axis", axes, Ops.to_int op)
  | Func_name s -> Hashtbl.hash ("fn", s)

(** Create a UOp. Hash-consed: identical (op, dtype, src, arg) returns same node. *)
let create ?(arg=No_arg) op dtype src =
  let dtype_hash = Hashtbl.hash dtype in
  let src_ids = List.map (fun (u : t) -> u.id) src in
  let arg_hash = hash_arg arg in
  let key = (op, dtype_hash lxor arg_hash, src_ids, arg_hash) in
  match Hashtbl.find_opt cache key with
  | Some u when u.op = op && u.dtype = dtype && u.arg = arg
    && List.length u.src = List.length src
    && List.for_all2 (fun a b -> a.id = b.id) u.src src -> u
  | _ ->
    let id = !next_id in
    incr next_id;
    let u = { op; dtype; src; arg; id } in
    Hashtbl.replace cache key u;
    u

(** Convenience constructors *)
let const_int dt v = create ~arg:(Int_arg v) Ops.CONST dt []
let const_float dt v = create ~arg:(Float_arg v) Ops.CONST dt []

let const dt (v : float) =
  match dt with
  | Dtype.Scalar Dtype.Bool -> const_int dt (if v <> 0.0 then 1 else 0)
  | Dtype.Scalar s when Dtype.is_int s -> const_int dt (Float.to_int v)
  | _ -> const_float dt v

(** Variable: a DEFINE_VAR with name, min, max *)
let variable name vmin vmax =
  create ~arg:(String_arg name) Ops.DEFINE_VAR Dtype.int32
    [const_int Dtype.int32 vmin; const_int Dtype.int32 vmax]

(** SINK combinator: merges multiple stores *)
let sink ?name srcs =
  let arg = match name with Some n -> Func_name n | None -> No_arg in
  create ~arg Ops.SINK Dtype.void srcs

(** Param: a kernel parameter *)
let param idx dtype = create ~arg:(Int_arg idx) Ops.PARAM dtype []

(** RANGE: a loop dimension *)
let range bound axis_arg =
  create ~arg:(Tuple_int axis_arg) Ops.RANGE Dtype.int32 [bound]

(** END: close a range *)
let end_ range_uop = create Ops.END Dtype.void [range_uop]

(** INDEX: pointer arithmetic — result is the same pointer type as the buffer *)
let index buf idx = create Ops.INDEX buf.dtype [buf; idx]

(** LOAD from an index *)
let load bidx =
  let base_dt = match bidx.dtype with
    | Dtype.Ptr (b, _, _) -> b
    | dt -> dt
  in
  create Ops.LOAD base_dt [bidx]

(** STORE to an index *)
let store bidx value = create Ops.STORE Dtype.void [bidx; value]

(** ALU operations *)
let alu op dtype srcs = create op dtype srcs

let add a b = alu Ops.ADD a.dtype [a; b]
let mul a b = alu Ops.MUL a.dtype [a; b]
let sub a b = alu Ops.SUB a.dtype [a; b]
let neg a = alu Ops.NEG a.dtype [a]
let idiv a b = alu Ops.IDIV a.dtype [a; b]
let max_ a b = alu Ops.MAX a.dtype [a; b]
let mod_ a b = alu Ops.MOD a.dtype [a; b]
let cmplt a b = alu Ops.CMPLT Dtype.bool [a; b]
let cmpne a b = alu Ops.CMPNE Dtype.bool [a; b]
let cmpeq a b = alu Ops.CMPEQ Dtype.bool [a; b]
let where_ cond t f = alu Ops.WHERE t.dtype [cond; t; f]
let exp2 a = alu Ops.EXP2 a.dtype [a]
let log2 a = alu Ops.LOG2 a.dtype [a]
let sin_ a = alu Ops.SIN a.dtype [a]
let sqrt_ a = alu Ops.SQRT a.dtype [a]
let recip a = alu Ops.RECIPROCAL a.dtype [a]
let trunc a = alu Ops.TRUNC a.dtype [a]

let shl a b = alu Ops.SHL a.dtype [a; b]
let shr a b = alu Ops.SHR a.dtype [a; b]
let bit_and a b = alu Ops.AND a.dtype [a; b]
let bit_or a b = alu Ops.OR a.dtype [a; b]
let bit_xor a b = alu Ops.XOR a.dtype [a; b]

(** CAST to a new dtype *)
let cast dt u = create Ops.CAST dt [u]
let bitcast dt u = create Ops.BITCAST dt [u]

(** Movement ops — these build the lazy tensor graph *)
let reshape u shape = create ~arg:(Shape shape) Ops.RESHAPE u.dtype [u]
let expand u shape = create ~arg:(Shape shape) Ops.EXPAND u.dtype [u]
let permute u axes = create ~arg:(Int_list axes) Ops.PERMUTE u.dtype [u]
let pad u padding = create ~arg:(Pad_arg padding) Ops.PAD u.dtype [u]
let shrink u bounds = create ~arg:(Pad_arg bounds) Ops.SHRINK u.dtype [u]
let flip u axes = create ~arg:(Int_list axes) Ops.FLIP u.dtype [u]

(** Reduce — stores (axes, reduce_op, src_shape) in the arg *)
let reduce_axis ?(src_shape=[]) u op axes =
  create ~arg:(Axis_arg (axes, op, src_shape)) Ops.REDUCE_AXIS u.dtype [u]

(** CONTIGUOUS *)
let contiguous u = create Ops.CONTIGUOUS u.dtype [u]

(** DETACH — stops gradient flow *)
let detach u = create Ops.DETACH u.dtype [u]

(** COPY *)
let copy u device = create ~arg:(String_arg device) Ops.COPY u.dtype [u]

(** BUFFER *)
let buffer idx dtype = create ~arg:(Int_arg idx) Ops.BUFFER dtype []

(** SPECIAL (GPU dims) *)
let special name bound =
  create ~arg:(String_arg name) Ops.SPECIAL Dtype.int32 [bound]

(** DEFINE_LOCAL (shared memory) *)
let define_local dt size =
  create ~arg:(Int_arg size) Ops.DEFINE_LOCAL (Dtype.ptr ~addrspace:Local dt) []

(** BARRIER *)
let barrier deps = create Ops.BARRIER Dtype.void deps

(** VECTORIZE *)
let vectorize srcs =
  match srcs with
  | [] -> failwith "vectorize requires at least one source"
  | [s] -> s
  | s :: _ ->
    let scalar = Dtype.scalar_of s.dtype in
    let dt = Dtype.vec scalar (List.length srcs) in
    create Ops.VECTORIZE dt srcs

(** GEP: get element pointer / vector element access *)
let gep u idx =
  let scalar_dt = match u.dtype with
    | Dtype.Vec (s, _) -> Dtype.Scalar s
    | dt -> dt
  in
  create ~arg:(Int_arg idx) Ops.GEP scalar_dt [u]

(** IF *)
let if_ cond = create Ops.IF Dtype.void [cond]
let endif if_uop = create Ops.ENDIF Dtype.void [if_uop]

(** Topological sort — iterative to avoid stack overflow.
    Returns UOps in dependency order (sources before consumers).
    Matches tinygrad's UOp.toposort(). *)
let toposort roots =
  let visited = Hashtbl.create 256 in
  let result = ref [] in
  (* iterative DFS using explicit stack *)
  let stack = Stack.create () in
  List.iter (fun r -> Stack.push (r, false) stack) (List.rev roots);
  while not (Stack.is_empty stack) do
    let (node, processed) = Stack.pop stack in
    if processed then begin
      if not (Hashtbl.mem visited node.id) then begin
        Hashtbl.add visited node.id ();
        result := node :: !result
      end
    end else if not (Hashtbl.mem visited node.id) then begin
      Stack.push (node, true) stack;
      List.iter (fun s -> if not (Hashtbl.mem visited s.id) then Stack.push (s, false) stack) (List.rev node.src)
    end
  done;
  List.rev !result

(** Topological sort of a single UOp *)
let toposort1 u = toposort [u]

(** Print a UOp for debugging *)
let rec pp_uop u =
  let src_str = match u.src with
    | [] -> ""
    | srcs -> Printf.sprintf " src=[%s]" (String.concat ", " (List.map (fun s -> Printf.sprintf "%%%d" s.id) srcs))
  in
  let arg_str = match u.arg with
    | No_arg -> ""
    | Int_arg i -> Printf.sprintf " arg=%d" i
    | Float_arg f -> Printf.sprintf " arg=%g" f
    | String_arg s -> Printf.sprintf " arg=%S" s
    | Int_list l -> Printf.sprintf " arg=[%s]" (String.concat "," (List.map string_of_int l))
    | Pair_int (a, b) -> Printf.sprintf " arg=(%d,%d)" a b
    | Tuple_int l -> Printf.sprintf " arg=(%s)" (String.concat "," (List.map string_of_int l))
    | Shape l -> Printf.sprintf " shape=(%s)" (String.concat "," (List.map string_of_int l))
    | Pad_arg l -> Printf.sprintf " pad=[%s]" (String.concat ";" (List.map (fun (a,b) -> Printf.sprintf "%d,%d" a b) l))
    | Axis_arg (axes, op, _) -> Printf.sprintf " axes=(%s,%s)" (String.concat "," (List.map string_of_int axes)) (Ops.to_string op)
    | Func_name s -> Printf.sprintf " fn=%S" s
  in
  Printf.sprintf "%%%d = %s %s%s%s" u.id (Ops.to_string u.op) (Dtype.to_string u.dtype) src_str arg_str

and print_uops uops =
  List.iter (fun u -> Printf.printf "%s\n" (pp_uop u)) uops

(** Comparison by id for use in sets/maps *)
let compare a b = Int.compare a.id b.id
let equal a b = a.id = b.id

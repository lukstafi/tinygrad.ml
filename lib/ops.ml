(** Operation types for the UOp IR, ported from tinygrad/uop/__init__.py.
    The order here controls toposort ordering — lower variants sort first. *)

type t =
  (* ** 1 -- defines/special ** *)
  | DEFINE_VAR | BIND
  | SPECIAL
  | DEFINE_LOCAL | DEFINE_REG
  (* ** 2 -- non-op uops ** *)
  | NOOP
  | PARAM | CALL
  | PROGRAM | LINEAR | SOURCE | BINARY
  | SINK | AFTER | GROUP
  | GEP | VECTORIZE
  (* ** 3 -- load/store ** *)
  | INDEX
  | LOAD | STORE
  (* ** 4 -- math ** *)
  | WMMA
  (* UnaryOps *)
  | CAST | BITCAST | EXP2 | LOG2 | SIN
  | SQRT | RECIPROCAL | NEG | TRUNC
  (* BinaryOps *)
  | ADD | MUL | SHL | SHR | IDIV | MAX | MOD
  | CMPLT | CMPNE | CMPEQ
  | XOR | OR | AND
  | THREEFRY | SUB | FDIV | POW
  (* TernaryOps *)
  | WHERE | MULACC
  (* ** 5 -- control flow / consts / custom ** *)
  | BARRIER | RANGE | IF | END | ENDIF
  | VCONST | CONST
  | CUSTOM | CUSTOMI
  (* ** 6 -- ops that don't exist in programs ** *)
  | UNIQUE | DEVICE | ASSIGN
  | CONTIGUOUS | CONTIGUOUS_BACKWARD | DETACH
  | BUFFERIZE | COPY | BUFFER | BUFFER_VIEW
  | RESHAPE | PERMUTE | EXPAND | PAD | SHRINK | FLIP
  | MULTI
  | REDUCE_AXIS | REDUCE
  | UNROLL | CONTRACT

(** String representation for rendering *)
let to_string = function
  | DEFINE_VAR -> "DEFINE_VAR" | BIND -> "BIND"
  | SPECIAL -> "SPECIAL" | DEFINE_LOCAL -> "DEFINE_LOCAL" | DEFINE_REG -> "DEFINE_REG"
  | NOOP -> "NOOP" | PARAM -> "PARAM" | CALL -> "CALL"
  | PROGRAM -> "PROGRAM" | LINEAR -> "LINEAR" | SOURCE -> "SOURCE" | BINARY -> "BINARY"
  | SINK -> "SINK" | AFTER -> "AFTER" | GROUP -> "GROUP"
  | GEP -> "GEP" | VECTORIZE -> "VECTORIZE"
  | INDEX -> "INDEX" | LOAD -> "LOAD" | STORE -> "STORE"
  | WMMA -> "WMMA"
  | CAST -> "CAST" | BITCAST -> "BITCAST" | EXP2 -> "EXP2" | LOG2 -> "LOG2" | SIN -> "SIN"
  | SQRT -> "SQRT" | RECIPROCAL -> "RECIPROCAL" | NEG -> "NEG" | TRUNC -> "TRUNC"
  | ADD -> "ADD" | MUL -> "MUL" | SHL -> "SHL" | SHR -> "SHR" | IDIV -> "IDIV"
  | MAX -> "MAX" | MOD -> "MOD"
  | CMPLT -> "CMPLT" | CMPNE -> "CMPNE" | CMPEQ -> "CMPEQ"
  | XOR -> "XOR" | OR -> "OR" | AND -> "AND"
  | THREEFRY -> "THREEFRY" | SUB -> "SUB" | FDIV -> "FDIV" | POW -> "POW"
  | WHERE -> "WHERE" | MULACC -> "MULACC"
  | BARRIER -> "BARRIER" | RANGE -> "RANGE" | IF -> "IF" | END -> "END" | ENDIF -> "ENDIF"
  | VCONST -> "VCONST" | CONST -> "CONST"
  | CUSTOM -> "CUSTOM" | CUSTOMI -> "CUSTOMI"
  | UNIQUE -> "UNIQUE" | DEVICE -> "DEVICE" | ASSIGN -> "ASSIGN"
  | CONTIGUOUS -> "CONTIGUOUS" | CONTIGUOUS_BACKWARD -> "CONTIGUOUS_BACKWARD" | DETACH -> "DETACH"
  | BUFFERIZE -> "BUFFERIZE" | COPY -> "COPY" | BUFFER -> "BUFFER" | BUFFER_VIEW -> "BUFFER_VIEW"
  | RESHAPE -> "RESHAPE" | PERMUTE -> "PERMUTE" | EXPAND -> "EXPAND"
  | PAD -> "PAD" | SHRINK -> "SHRINK" | FLIP -> "FLIP"
  | MULTI -> "MULTI"
  | REDUCE_AXIS -> "REDUCE_AXIS" | REDUCE -> "REDUCE"
  | UNROLL -> "UNROLL" | CONTRACT -> "CONTRACT"

(** Integer encoding for comparison / ordering in toposort *)
let to_int = function
  | DEFINE_VAR -> 0 | BIND -> 1 | SPECIAL -> 2
  | DEFINE_LOCAL -> 3 | DEFINE_REG -> 4
  | NOOP -> 5 | PARAM -> 6 | CALL -> 7
  | PROGRAM -> 8 | LINEAR -> 9 | SOURCE -> 10 | BINARY -> 11
  | SINK -> 12 | AFTER -> 13 | GROUP -> 14
  | GEP -> 15 | VECTORIZE -> 16
  | INDEX -> 17 | LOAD -> 18 | STORE -> 19
  | WMMA -> 20
  | CAST -> 21 | BITCAST -> 22 | EXP2 -> 23 | LOG2 -> 24 | SIN -> 25
  | SQRT -> 26 | RECIPROCAL -> 27 | NEG -> 28 | TRUNC -> 29
  | ADD -> 30 | MUL -> 31 | SHL -> 32 | SHR -> 33 | IDIV -> 34
  | MAX -> 35 | MOD -> 36
  | CMPLT -> 37 | CMPNE -> 38 | CMPEQ -> 39
  | XOR -> 40 | OR -> 41 | AND -> 42
  | THREEFRY -> 43 | SUB -> 44 | FDIV -> 45 | POW -> 46
  | WHERE -> 47 | MULACC -> 48
  | BARRIER -> 49 | RANGE -> 50 | IF -> 51 | END -> 52 | ENDIF -> 53
  | VCONST -> 54 | CONST -> 55
  | CUSTOM -> 56 | CUSTOMI -> 57
  | UNIQUE -> 58 | DEVICE -> 59 | ASSIGN -> 60
  | CONTIGUOUS -> 61 | CONTIGUOUS_BACKWARD -> 62 | DETACH -> 63
  | BUFFERIZE -> 64 | COPY -> 65 | BUFFER -> 66 | BUFFER_VIEW -> 67
  | RESHAPE -> 68 | PERMUTE -> 69 | EXPAND -> 70
  | PAD -> 71 | SHRINK -> 72 | FLIP -> 73
  | MULTI -> 74
  | REDUCE_AXIS -> 75 | REDUCE -> 76
  | UNROLL -> 77 | CONTRACT -> 78

let compare a b = Int.compare (to_int a) (to_int b)

(** Op group sets, ported from GroupOp *)
module Group = struct
  let unary = [EXP2; LOG2; SIN; SQRT; RECIPROCAL; NEG; TRUNC]
  let binary = [ADD; MUL; IDIV; MAX; MOD; CMPLT; CMPNE; CMPEQ; XOR; SHL; SHR; OR; AND; THREEFRY; SUB; FDIV; POW]
  let ternary = [WHERE; MULACC]
  let alu = unary @ binary @ ternary
  let elementwise = alu @ [CAST; BITCAST]
  let movement = [RESHAPE; EXPAND; PERMUTE; PAD; SHRINK; FLIP]
  let commutative = [ADD; MUL; MAX; CMPNE; CMPEQ; XOR; AND; OR]
  let comparison = [CMPLT; CMPNE; CMPEQ]

  let mem op l = List.mem op l
  let is_alu op = mem op alu
  let is_unary op = mem op unary
  let is_binary op = mem op binary
  let is_ternary op = mem op ternary
  let is_elementwise op = mem op elementwise
  let is_movement op = mem op movement
  let is_commutative op = mem op commutative
  let is_comparison op = mem op comparison

  (** Single-source view/wrapper ops: movement ops + CAST + CONTIGUOUS.
      These ops have exactly one input and don't change the computation semantics
      in a way that affects broadcast shape inference. Used by scheduler to walk
      through wrapper chains when looking for RESHAPE nodes. *)
  let view_wrapper = [RESHAPE; EXPAND; PERMUTE; PAD; SHRINK; FLIP; CAST; CONTIGUOUS]
  let is_view_wrapper op = mem op view_wrapper
end

(** Identity element for reduction ops *)
let identity_element op (dt : Dtype.t) =
  match op with
  | ADD -> 0.0
  | MUL -> 1.0
  | MAX -> (match dt with Scalar s | Vec (s, _) -> Dtype.min_val s | _ -> Float.neg_infinity)
  | _ -> failwith (Printf.sprintf "no identity element for %s" (to_string op))

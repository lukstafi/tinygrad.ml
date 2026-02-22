type binop = Add | Sub | Mul | Lt | Eq | Ne

type unop = Neg | Sqrt | Reciprocal | Exp2 | Log2 | Sin

type reduce_op = Sum | Max

type expr =
  | Input of int
  | Const of float
  | Binop of binop * expr * expr
  | Unop of unop * expr
  | Cast of Dtype.scalar * expr
  | Where of expr * expr * expr

let binop_to_name = function
  | Add -> "add"
  | Sub -> "sub"
  | Mul -> "mul"
  | Lt -> "lt"
  | Eq -> "eq"
  | Ne -> "ne"

let binop_to_symbol = function
  | Add -> "+"
  | Sub -> "-"
  | Mul -> "*"
  | Lt -> "<"
  | Eq -> "=="
  | Ne -> "!="

let unop_to_name = function
  | Neg -> "neg"
  | Sqrt -> "sqrt"
  | Reciprocal -> "reciprocal"
  | Exp2 -> "exp2"
  | Log2 -> "log2"
  | Sin -> "sin"

let reduce_op_to_name = function
  | Sum -> "sum"
  | Max -> "max"

let scalar_to_name = function
  | Dtype.F32 -> "f32"
  | Dtype.I32 -> "i32"
  | Dtype.Bool -> "bool"

let rec expr_to_key = function
  | Input i -> Printf.sprintf "i%d" i
  | Const c -> Printf.sprintf "c(%0.9g)" c
  | Binop (op, a, b) ->
      Printf.sprintf "%s(%s,%s)" (binop_to_name op) (expr_to_key a) (expr_to_key b)
  | Unop (op, x) ->
      Printf.sprintf "%s(%s)" (unop_to_name op) (expr_to_key x)
  | Cast (dtype, x) ->
      Printf.sprintf "cast_%s(%s)" (scalar_to_name dtype) (expr_to_key x)
  | Where (c, t, f) ->
      Printf.sprintf "where(%s,%s,%s)" (expr_to_key c) (expr_to_key t) (expr_to_key f)

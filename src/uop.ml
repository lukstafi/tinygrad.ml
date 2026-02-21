type binop = Add | Sub | Mul

type unop = Neg | Sqrt | Reciprocal

type reduce_op = Sum | Max

type expr =
  | Input of int
  | Const of float
  | Binop of binop * expr * expr
  | Unop of unop * expr

let binop_to_name = function
  | Add -> "add"
  | Sub -> "sub"
  | Mul -> "mul"

let binop_to_symbol = function
  | Add -> "+"
  | Sub -> "-"
  | Mul -> "*"

let unop_to_name = function
  | Neg -> "neg"
  | Sqrt -> "sqrt"
  | Reciprocal -> "reciprocal"

let reduce_op_to_name = function
  | Sum -> "sum"
  | Max -> "max"

let rec expr_to_key = function
  | Input i -> Printf.sprintf "i%d" i
  | Const c -> Printf.sprintf "c(%0.9g)" c
  | Binop (op, a, b) ->
      Printf.sprintf "%s(%s,%s)" (binop_to_name op) (expr_to_key a) (expr_to_key b)
  | Unop (op, x) ->
      Printf.sprintf "%s(%s)" (unop_to_name op) (expr_to_key x)

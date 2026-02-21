type binop = Add | Mul

type t =
  | Input of string
  | Binop of binop * t * t

let binop_to_name = function
  | Add -> "add"
  | Mul -> "mul"

let binop_to_symbol = function
  | Add -> "+"
  | Mul -> "*"

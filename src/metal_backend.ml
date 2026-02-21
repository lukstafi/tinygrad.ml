let device_name = "metal"

let available () = Error "metal backend unavailable (missing metal library at build/runtime)"

let run_expr ~expr:_ ~inputs:_ ~shape:_ =
  Error "metal backend unavailable (build tinygrad_ml with metal bindings)"

let run_binop ~op:_ ~a:_ ~b:_ =
  Error "metal backend unavailable (build tinygrad_ml with metal bindings)"

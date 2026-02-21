let device_name = "cuda"

let available () = Error "cuda backend unavailable (build without cudajit.cuda + cudajit.nvrtc)"

let run_expr ~expr:_ ~inputs:_ ~shape:_ =
  Error "cuda backend unavailable (install/build with cudajit.cuda + cudajit.nvrtc)"

let run_reduce ~op:_ ~expr:_ ~inputs:_ ~shape:_ =
  Error "cuda backend unavailable (install/build with cudajit.cuda + cudajit.nvrtc)"

let run_binop ~op:_ ~a:_ ~b:_ =
  Error "cuda backend unavailable (install/build with cudajit.cuda + cudajit.nvrtc)"

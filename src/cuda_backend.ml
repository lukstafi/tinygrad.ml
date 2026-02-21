let device_name = "cuda"

let available () = Error "cuda backend unavailable (missing cudajit libraries at build/runtime)"

let run_binop ~op:_ ~a:_ ~b:_ =
  Error "cuda backend unavailable (build tinygrad_ml with cudajit.cuda + cudajit.nvrtc)"

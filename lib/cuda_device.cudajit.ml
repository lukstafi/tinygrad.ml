(** CUDA device implementation — delegates to Cuda_backend when cudajit is available. *)

let device_name = Cuda_backend.device_name
let alloc = Cuda_backend.alloc
let free = Cuda_backend.free
let copyin = Cuda_backend.copyin
let copyout = Cuda_backend.copyout
let compile = Cuda_backend.compile
let exec = Cuda_backend.exec
let synchronize = Cuda_backend.synchronize
let is_available = Cuda_backend.is_available

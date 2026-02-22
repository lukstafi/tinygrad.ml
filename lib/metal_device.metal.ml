(** Metal device implementation — delegates to Metal_backend when the metal package is available. *)

let device_name = "METAL"
let alloc = Metal_backend.alloc
let free = Metal_backend.free
let copyin = Metal_backend.copyin
let copyout = Metal_backend.copyout
let compile = Metal_backend.compile
let exec = Metal_backend.exec
let synchronize = Metal_backend.synchronize
let is_available = true

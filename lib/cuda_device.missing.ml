(** CUDA device stub — used when the cudajit package is not installed. *)

let device_name = "CUDA"
let _unavail () = failwith "CUDA backend: cudajit package not installed (opam install cudajit)"
let alloc _nbytes = _unavail ()
let free _ptr = _unavail ()
let copyin _dst _src = _unavail ()
let copyout _dst _src = _unavail ()
let compile _name _src = _unavail ()
let exec _name _bin _ptrs _vals = _unavail ()
let synchronize () = ()
let is_available = false

(** Metal device stub — used when the metal package is not installed. *)

let device_name = "METAL"
let _unavail () = failwith "Metal backend: metal package not installed (opam install metal)"
let alloc _nbytes = _unavail ()
let free _ptr = _unavail ()
let copyin _dst _src = _unavail ()
let copyout _dst _src = _unavail ()
let compile _name _src = _unavail ()
let exec _name _bin _ptrs _vals = _unavail ()
let synchronize () = ()
let is_available = false

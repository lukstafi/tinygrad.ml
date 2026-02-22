(** CUDA GPU backend for tinygrad_ml.
    Uses the OCaml `cudajit` package to compile PTX/CUDA kernels and execute them.
    Ported from tinygrad/runtime/ops_cuda.py (simplified).

    This is a placeholder — when cudajit is installed, this module should be
    filled in with real CUDA runtime bindings using Cuda and Nvrtc modules. *)

let device_name = "CUDA"
let _unavail () = failwith "CUDA backend: cudajit not installed (opam install cudajit for GPU execution)"
let alloc (_nbytes : int) : nativeint = _unavail ()
let free (_ptr : nativeint) : unit = _unavail ()
let copyin (_dst : nativeint) (_src : (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t) : unit = _unavail ()
let copyout (_dst : (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t) (_src : nativeint) : unit = _unavail ()
let compile (_name : string) (_src : string) : string = _unavail ()
let exec (_name : string) (_bin : string) (_ptrs : nativeint list) (_vals : int list) : unit = _unavail ()
let synchronize () = ()
let is_available = false  (* placeholder: set true once real CUDA bindings are implemented *)

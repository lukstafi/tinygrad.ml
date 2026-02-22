(** Device abstraction and Buffer type.
    Ported from tinygrad/device.py.

    Provides a uniform interface for allocating memory, compiling programs,
    and executing kernels across CPU, CUDA, and Metal backends. *)

open Ctypes
open Foreign

(** A buffer: device memory with metadata *)
type buffer = {
  device: string;
  size: int;           (** number of elements *)
  dtype: Dtype.t;
  mutable ptr: nativeint;  (** raw pointer to allocated memory *)
  nbytes: int;
}

(** Create an unallocated buffer *)
let make_buffer ~device ~size ~dtype =
  { device; size; dtype; ptr = Nativeint.zero; nbytes = size * Dtype.itemsize dtype }

(** Device interface — each backend implements this *)
module type Backend = sig
  val device_name : string

  (** Allocate device memory, returns raw pointer *)
  val alloc : int -> nativeint

  (** Free device memory *)
  val free : nativeint -> unit

  (** Copy from host Bigarray to device *)
  val copyin : nativeint -> (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t -> unit

  (** Copy from device to host Bigarray *)
  val copyout : (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t -> nativeint -> unit

  (** Compile source code, returns (so_path, binary_bytes) *)
  val compile : string -> string -> string

  (** Execute a compiled program *)
  val exec : string -> string -> nativeint list -> int list -> unit
    (** (kernel_name, so_path, buffer_ptrs, int_vals) *)

  (** Synchronize: wait for all pending operations *)
  val synchronize : unit -> unit
end

(** CPU backend — compiles C to shared library via clang, executes via dlopen *)
module CPU : Backend = struct
  let device_name = "CPU"

  (* Keep Bigarrays alive so the GC doesn't collect the backing memory
     while we still hold raw pointers to it. *)
  let ba_roots : (nativeint, (char, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t) Hashtbl.t = Hashtbl.create 64

  let alloc nbytes =
    let ba = Bigarray.Array1.create Bigarray.char Bigarray.c_layout nbytes in
    Bigarray.Array1.fill ba '\000';
    let p = bigarray_start array1 ba in
    let addr = raw_address_of_ptr (to_voidp p) in
    Hashtbl.replace ba_roots addr ba;
    addr

  let free ptr =
    Hashtbl.remove ba_roots ptr

  let copyin dst_ptr src =
    let n = Bigarray.Array1.dim src in
    let dst_p = coerce (ptr void) (ptr char) (ptr_of_raw_address dst_ptr) in
    let dst_ba = bigarray_of_ptr array1 n Bigarray.char dst_p in
    Bigarray.Array1.blit src dst_ba

  let copyout dst src_ptr =
    let n = Bigarray.Array1.dim dst in
    let src_p = coerce (ptr void) (ptr char) (ptr_of_raw_address src_ptr) in
    let src_ba = bigarray_of_ptr array1 n Bigarray.char src_p in
    Bigarray.Array1.blit src_ba dst

  let compile_counter = ref 0

  (* Cache compiled .so paths by source code hash to avoid recompiling identical kernels *)
  let compile_cache : (string, string) Hashtbl.t = Hashtbl.create 32

  let compile name src =
    match Hashtbl.find_opt compile_cache src with
    | Some so_file -> so_file
    | None ->
      let tmpdir = Filename.get_temp_dir_name () in
      let uid = !compile_counter in
      incr compile_counter;
      let src_file = Filename.concat tmpdir (Printf.sprintf "%s_%d.c" name uid) in
      let so_file = Filename.concat tmpdir (Printf.sprintf "%s_%d.so" name uid) in
      let oc = open_out src_file in
      output_string oc src;
      close_out oc;
      let cmd = Printf.sprintf "cc -shared -O2 -fPIC -o %s %s -lm 2>&1" so_file src_file in
      let ic = Unix.open_process_in cmd in
      let output = Buffer.create 256 in
      (try while true do Buffer.add_char output (input_char ic) done with End_of_file -> ());
      let status = Unix.close_process_in ic in
      (match status with
       | Unix.WEXITED 0 -> ()
       | _ -> failwith (Printf.sprintf "Compilation failed: %s\n%s" cmd (Buffer.contents output)));
      Hashtbl.replace compile_cache src so_file;
      so_file

  (* Cache for loaded libraries *)
  let lib_cache : (string, Dl.library) Hashtbl.t = Hashtbl.create 16

  let get_lib so_path =
    match Hashtbl.find_opt lib_cache so_path with
    | Some lib -> lib
    | None ->
      let lib = Dl.dlopen ~filename:so_path ~flags:[Dl.RTLD_NOW] in
      Hashtbl.replace lib_cache so_path lib;
      lib

  let exec kernel_name so_path buf_ptrs int_vals =
    let lib = get_lib so_path in
    (* Build the function type dynamically based on number of args *)
    let n_bufs = List.length buf_ptrs in
    let n_vals = List.length int_vals in
    (* All buffer args are pointer-sized (uint64), int args are int32 *)
    (* We build a C function call dynamically *)
    ignore lib; ignore kernel_name;
    (* Use Foreign to look up the function *)
    let fptr_raw = Dl.dlsym ~handle:lib ~symbol:kernel_name in
    let fptr = ptr_of_raw_address fptr_raw in
    (* Call via raw function pointer, dispatch on arg count *)
    match n_bufs, n_vals with
    | 1, 0 ->
      let f = coerce (ptr void) (funptr (ptr void @-> returning void)) fptr in
      f (ptr_of_raw_address (List.nth buf_ptrs 0))
    | 2, 0 ->
      let f = coerce (ptr void) (funptr (ptr void @-> ptr void @-> returning void)) fptr in
      f (ptr_of_raw_address (List.nth buf_ptrs 0))
        (ptr_of_raw_address (List.nth buf_ptrs 1))
    | 3, 0 ->
      let f = coerce (ptr void) (funptr (ptr void @-> ptr void @-> ptr void @-> returning void)) fptr in
      f (ptr_of_raw_address (List.nth buf_ptrs 0))
        (ptr_of_raw_address (List.nth buf_ptrs 1))
        (ptr_of_raw_address (List.nth buf_ptrs 2))
    | 2, 1 ->
      let f = coerce (ptr void) (funptr (ptr void @-> ptr void @-> int @-> returning void)) fptr in
      f (ptr_of_raw_address (List.nth buf_ptrs 0))
        (ptr_of_raw_address (List.nth buf_ptrs 1))
        (List.nth int_vals 0)
    | 3, 1 ->
      let f = coerce (ptr void) (funptr (ptr void @-> ptr void @-> ptr void @-> int @-> returning void)) fptr in
      f (ptr_of_raw_address (List.nth buf_ptrs 0))
        (ptr_of_raw_address (List.nth buf_ptrs 1))
        (ptr_of_raw_address (List.nth buf_ptrs 2))
        (List.nth int_vals 0)
    | 4, 0 ->
      let f = coerce (ptr void) (funptr (ptr void @-> ptr void @-> ptr void @-> ptr void @-> returning void)) fptr in
      f (ptr_of_raw_address (List.nth buf_ptrs 0))
        (ptr_of_raw_address (List.nth buf_ptrs 1))
        (ptr_of_raw_address (List.nth buf_ptrs 2))
        (ptr_of_raw_address (List.nth buf_ptrs 3))
    | 4, 1 ->
      let f = coerce (ptr void) (funptr (ptr void @-> ptr void @-> ptr void @-> ptr void @-> int @-> returning void)) fptr in
      f (ptr_of_raw_address (List.nth buf_ptrs 0))
        (ptr_of_raw_address (List.nth buf_ptrs 1))
        (ptr_of_raw_address (List.nth buf_ptrs 2))
        (ptr_of_raw_address (List.nth buf_ptrs 3))
        (List.nth int_vals 0)
    | 5, 0 ->
      let f = coerce (ptr void) (funptr (ptr void @-> ptr void @-> ptr void @-> ptr void @-> ptr void @-> returning void)) fptr in
      f (ptr_of_raw_address (List.nth buf_ptrs 0))
        (ptr_of_raw_address (List.nth buf_ptrs 1))
        (ptr_of_raw_address (List.nth buf_ptrs 2))
        (ptr_of_raw_address (List.nth buf_ptrs 3))
        (ptr_of_raw_address (List.nth buf_ptrs 4))
    | 6, 0 ->
      let f = coerce (ptr void) (funptr (ptr void @-> ptr void @-> ptr void @-> ptr void @-> ptr void @-> ptr void @-> returning void)) fptr in
      f (ptr_of_raw_address (List.nth buf_ptrs 0))
        (ptr_of_raw_address (List.nth buf_ptrs 1))
        (ptr_of_raw_address (List.nth buf_ptrs 2))
        (ptr_of_raw_address (List.nth buf_ptrs 3))
        (ptr_of_raw_address (List.nth buf_ptrs 4))
        (ptr_of_raw_address (List.nth buf_ptrs 5))
    | 7, 0 ->
      let f = coerce (ptr void) (funptr (ptr void @-> ptr void @-> ptr void @-> ptr void @-> ptr void @-> ptr void @-> ptr void @-> returning void)) fptr in
      f (ptr_of_raw_address (List.nth buf_ptrs 0))
        (ptr_of_raw_address (List.nth buf_ptrs 1))
        (ptr_of_raw_address (List.nth buf_ptrs 2))
        (ptr_of_raw_address (List.nth buf_ptrs 3))
        (ptr_of_raw_address (List.nth buf_ptrs 4))
        (ptr_of_raw_address (List.nth buf_ptrs 5))
        (ptr_of_raw_address (List.nth buf_ptrs 6))
    | 8, 0 ->
      let f = coerce (ptr void) (funptr (ptr void @-> ptr void @-> ptr void @-> ptr void @-> ptr void @-> ptr void @-> ptr void @-> ptr void @-> returning void)) fptr in
      f (ptr_of_raw_address (List.nth buf_ptrs 0))
        (ptr_of_raw_address (List.nth buf_ptrs 1))
        (ptr_of_raw_address (List.nth buf_ptrs 2))
        (ptr_of_raw_address (List.nth buf_ptrs 3))
        (ptr_of_raw_address (List.nth buf_ptrs 4))
        (ptr_of_raw_address (List.nth buf_ptrs 5))
        (ptr_of_raw_address (List.nth buf_ptrs 6))
        (ptr_of_raw_address (List.nth buf_ptrs 7))
    | _ ->
      failwith (Printf.sprintf "CPU.exec: unsupported arg count: %d bufs, %d vals" n_bufs n_vals)

  let synchronize () = ()
end

(** Metal backend — compiles MSL to Metal library, executes via Metal compute pipeline.
    Available only when the [metal] opam package is installed; otherwise stubs that
    raise at runtime are selected by dune. *)
module Metal : Backend = struct
  let device_name = Metal_device.device_name
  let alloc = Metal_device.alloc
  let free = Metal_device.free
  let copyin = Metal_device.copyin
  let copyout = Metal_device.copyout
  let compile = Metal_device.compile
  let exec = Metal_device.exec
  let synchronize = Metal_device.synchronize
end

(** CUDA backend — compiles CUDA/PTX kernels via nvrtc, executes on GPU.
    Available only when the [cudajit] opam package is installed; otherwise stubs
    that raise at runtime are selected by dune. *)
module CUDA : Backend = struct
  let device_name = Cuda_device.device_name
  let alloc = Cuda_device.alloc
  let free = Cuda_device.free
  let copyin = Cuda_device.copyin
  let copyout = Cuda_device.copyout
  let compile = Cuda_device.compile
  let exec = Cuda_device.exec
  let synchronize = Cuda_device.synchronize
end

(** Get a backend module by device name *)
let get_backend device =
  match String.uppercase_ascii device with
  | "CPU" -> (module CPU : Backend)
  | "METAL" -> (module Metal : Backend)
  | "CUDA" -> (module CUDA : Backend)
  | _ -> failwith (Printf.sprintf "Unknown device: %s" device)

(** Allocate a buffer on its device *)
let alloc_buffer (buf : buffer) =
  let module B = (val get_backend buf.device : Backend) in
  buf.ptr <- B.alloc buf.nbytes;
  buf

(** Copy float array into a buffer *)
let copyin_floats (buf : buffer) (data : float array) =
  let module B = (val get_backend buf.device : Backend) in
  let n = Array.length data in
  let nbytes = n * 4 in (* assuming float32 *)
  let ba = Bigarray.Array1.create Bigarray.char Bigarray.c_layout nbytes in
  for i = 0 to n - 1 do
    let bits = Int32.bits_of_float data.(i) in
    let p = i * 4 in
    Bigarray.Array1.set ba p (Char.chr (Int32.to_int (Int32.logand bits 0xFFl)));
    Bigarray.Array1.set ba (p+1) (Char.chr (Int32.to_int (Int32.logand (Int32.shift_right_logical bits 8) 0xFFl)));
    Bigarray.Array1.set ba (p+2) (Char.chr (Int32.to_int (Int32.logand (Int32.shift_right_logical bits 16) 0xFFl)));
    Bigarray.Array1.set ba (p+3) (Char.chr (Int32.to_int (Int32.logand (Int32.shift_right_logical bits 24) 0xFFl)));
  done;
  B.copyin buf.ptr ba

(** Copy buffer contents to float array *)
let copyout_floats (buf : buffer) : float array =
  let module B = (val get_backend buf.device : Backend) in
  let n = buf.size in
  let nbytes = n * 4 in (* assuming float32 *)
  let ba = Bigarray.Array1.create Bigarray.char Bigarray.c_layout nbytes in
  B.copyout ba buf.ptr;
  let result = Array.make n 0.0 in
  for i = 0 to n - 1 do
    let p = i * 4 in
    let b0 = Char.code (Bigarray.Array1.get ba p) in
    let b1 = Char.code (Bigarray.Array1.get ba (p+1)) in
    let b2 = Char.code (Bigarray.Array1.get ba (p+2)) in
    let b3 = Char.code (Bigarray.Array1.get ba (p+3)) in
    let bits = Int32.logor
      (Int32.logor (Int32.of_int b0) (Int32.shift_left (Int32.of_int b1) 8))
      (Int32.logor (Int32.shift_left (Int32.of_int b2) 16) (Int32.shift_left (Int32.of_int b3) 24)) in
    result.(i) <- Int32.float_of_bits bits
  done;
  result

(** Synchronize a device *)
let synchronize device =
  let module B = (val get_backend device : Backend) in
  B.synchronize ()

(** Check if a backend is available (package installed) *)
let is_available device =
  match String.uppercase_ascii device with
  | "CPU" -> true
  | "METAL" -> Metal_device.is_available
  | "CUDA" -> Cuda_device.is_available
  | _ -> false

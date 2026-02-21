open Ctypes
open Foreign

let device_name = "cpu-c"

type kernel_fn =
  (float Ctypes.ptr -> float Ctypes.ptr -> float Ctypes.ptr -> int -> unit)

type kernels = {
  add : kernel_fn;
  mul : kernel_fn;
}

let kernels_cache : kernels option ref = ref None

let build_shared_library () =
  let source =
    {|
#include <stdint.h>

void tg_add(float *out, const float *a, const float *b, int n) {
  for (int i = 0; i < n; i++) out[i] = a[i] + b[i];
}

void tg_mul(float *out, const float *a, const float *b, int n) {
  for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
}
|}
  in
  let c_file = Filename.temp_file "tinygrad_ml_cpu" ".c" in
  let so_file = c_file ^ ".so" in
  let oc = open_out_bin c_file in
  output_string oc source;
  close_out oc;
  let cc = Option.value (Sys.getenv_opt "CC") ~default:"cc" in
  let cmd =
    Printf.sprintf "%s -O3 -std=c99 -shared -fPIC -o %s %s"
      (Filename.quote cc)
      (Filename.quote so_file)
      (Filename.quote c_file)
  in
  match Sys.command cmd with
  | 0 -> so_file
  | code -> failwith (Printf.sprintf "failed to compile CPU backend C kernel (%d)" code)

let load_kernels () =
  let so_file = build_shared_library () in
  let handle = Dl.dlopen ~filename:so_file ~flags:[ Dl.RTLD_NOW ] in
  let kernel_typ = ptr float @-> ptr float @-> ptr float @-> int @-> returning void in
  {
    add = foreign ~from:handle "tg_add" kernel_typ;
    mul = foreign ~from:handle "tg_mul" kernel_typ;
  }

let ensure_kernels () =
  match !kernels_cache with
  | Some k -> k
  | None ->
      let k = load_kernels () in
      kernels_cache := Some k;
      k

let available () =
  try
    ignore (ensure_kernels ());
    Ok ()
  with exn -> Error (Printexc.to_string exn)

let run_binop ~(op : Uop.binop) ~(a : Buffer.t) ~(b : Buffer.t) =
  if not (Buffer.same_shape a b) then
    Error
      (Printf.sprintf "shape mismatch %s vs %s" (Buffer.pp_shape a.shape) (Buffer.pp_shape b.shape))
  else
    try
      let kernels = ensure_kernels () in
      let fn =
        match op with
        | Uop.Add -> kernels.add
        | Uop.Mul -> kernels.mul
      in
      let out = Buffer.create a.shape in
      let n = Bigarray.Array1.dim a.data in
      let out_ptr = Ctypes.bigarray_start Ctypes.array1 out.data in
      let a_ptr = Ctypes.bigarray_start Ctypes.array1 a.data in
      let b_ptr = Ctypes.bigarray_start Ctypes.array1 b.data in
      fn out_ptr a_ptr b_ptr n;
      Ok out
    with exn -> Error (Printexc.to_string exn)

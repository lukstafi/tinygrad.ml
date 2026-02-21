let render ~(op : Uop.binop) ~(length : int) : Program_spec.t =
  let fname = "tg_" ^ Uop.binop_to_name op in
  let expr =
    match op with
    | Uop.Add -> "a[i] + b[i]"
    | Uop.Mul -> "a[i] * b[i]"
  in
  let src =
    Printf.sprintf
      {|
#include <metal_stdlib>
using namespace metal;

kernel void %s(device float *out [[buffer(0)]],
               device const float *a [[buffer(1)]],
               device const float *b [[buffer(2)]],
               constant int &n [[buffer(3)]],
               uint gid [[thread_position_in_grid]]) {
  int i = int(gid);
  if (i < n) {
    out[i] = %s;
  }
}
|}
      fname expr
  in
  { Program_spec.function_name = fname; src; op; length }

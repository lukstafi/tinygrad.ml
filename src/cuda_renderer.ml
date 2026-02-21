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
extern "C" __global__ void %s(float *out, const float *a, const float *b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    out[i] = %s;
  }
}
|}
      fname expr
  in
  { Program_spec.function_name = fname; src; op; length }

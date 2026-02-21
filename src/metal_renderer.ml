let rec render_expr = function
  | Uop.Input i -> Printf.sprintf "in%d[i]" i
  | Uop.Const c -> Printf.sprintf "%gf" c
  | Uop.Binop (op, a, b) ->
      Printf.sprintf "(%s %s %s)" (render_expr a) (Uop.binop_to_symbol op) (render_expr b)
  | Uop.Unop (Uop.Neg, x) -> Printf.sprintf "(-(%s))" (render_expr x)
  | Uop.Unop (Uop.Sqrt, x) -> Printf.sprintf "sqrt(%s)" (render_expr x)
  | Uop.Unop (Uop.Reciprocal, x) -> Printf.sprintf "(1.0f/(%s))" (render_expr x)

let render_expr_kernel ~(expr : Uop.expr) ~(ninputs : int) ~(length : int) : Program_spec.t =
  let expression_key = Uop.expr_to_key expr in
  let digest = Digest.to_hex (Digest.string expression_key) in
  let function_name = "tg_expr_" ^ String.sub digest 0 16 in
  let input_args =
    List.init ninputs (fun i ->
        Printf.sprintf "device const float *in%d [[buffer(%d)]]" i (i + 1))
    |> String.concat ",\n               "
  in
  let arg_block =
    if ninputs = 0 then "constant int *n [[buffer(1)]],"
    else Printf.sprintf "%s,\n               constant int *n [[buffer(%d)]]," input_args (ninputs + 1)
  in
  let src =
    Printf.sprintf
      {|
#include <metal_stdlib>
using namespace metal;

kernel void %s(device float *out [[buffer(0)]],
               %s
               uint gid [[thread_position_in_grid]]) {
  int i = int(gid);
  if (i < *n) {
    out[i] = %s;
  }
}
|}
      function_name arg_block (render_expr expr)
  in
  { Program_spec.function_name; src; length; n_inputs = ninputs; expression_key }

let render ~(op : Uop.binop) ~(length : int) : Program_spec.t =
  render_expr_kernel ~expr:(Uop.Binop (op, Uop.Input 0, Uop.Input 1)) ~ninputs:2 ~length

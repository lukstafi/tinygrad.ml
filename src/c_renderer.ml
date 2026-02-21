let rec render_expr = function
  | Uop.Input i -> Printf.sprintf "in%d[i]" i
  | Uop.Const c -> Printf.sprintf "%gf" c
  | Uop.Binop (op, a, b) ->
      Printf.sprintf "(%s %s %s)" (render_expr a) (Uop.binop_to_symbol op) (render_expr b)
  | Uop.Unop (Uop.Neg, x) -> Printf.sprintf "(-(%s))" (render_expr x)
  | Uop.Unop (Uop.Sqrt, x) -> Printf.sprintf "sqrtf(%s)" (render_expr x)
  | Uop.Unop (Uop.Reciprocal, x) -> Printf.sprintf "(1.0f/(%s))" (render_expr x)

let render_expr_kernel ~(expr : Uop.expr) ~(ninputs : int) ~(length : int) : Program_spec.t =
  let expression_key = Uop.expr_to_key expr in
  let digest = Digest.to_hex (Digest.string expression_key) in
  let function_name = "tg_expr_" ^ String.sub digest 0 16 in
  let inputs =
    List.init ninputs (fun i -> Printf.sprintf "const float *in%d" i) |> String.concat ",\n                 "
  in
  let params =
    if ninputs = 0 then "float *out, int n"
    else Printf.sprintf "float *out,\n                 %s,\n                 int n" inputs
  in
  let src =
    Printf.sprintf
      {|
#include <math.h>
#include <stdint.h>

void %s(%s) {
  for (int i = 0; i < n; i++) {
    out[i] = %s;
  }
}
|}
      function_name params (render_expr expr)
  in
  { Program_spec.function_name; src; length; n_inputs = ninputs; expression_key }

let render ~(op : Uop.binop) ~(length : int) : Program_spec.t =
  render_expr_kernel ~expr:(Uop.Binop (op, Uop.Input 0, Uop.Input 1)) ~ninputs:2 ~length

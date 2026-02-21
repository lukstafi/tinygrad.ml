open Ctypes
open Foreign

let device_name = "cpu-c"

type compiled_kernel = {
  handle : Dl.library;
  function_name : string;
  n_inputs : int;
}

let kernel_cache : (string, compiled_kernel) Hashtbl.t = Hashtbl.create 64

let compiled_kernel_count () = Hashtbl.length kernel_cache
let reset_kernel_cache_for_tests () = Hashtbl.clear kernel_cache

let build_shared_library ~(source : string) =
  let c_file = Filename.temp_file "tinygrad_ml_cpu" ".c" in
  let so_file = c_file ^ ".so" in
  let oc = open_out_bin c_file in
  output_string oc source;
  close_out oc;
  let cc = Option.value (Sys.getenv_opt "CC") ~default:"cc" in
  let cmd =
    Printf.sprintf "%s -O3 -std=c99 -shared -fPIC -o %s %s -lm"
      (Filename.quote cc)
      (Filename.quote so_file)
      (Filename.quote c_file)
  in
  match Sys.command cmd with
  | 0 -> so_file
  | code -> failwith (Printf.sprintf "failed to compile CPU backend C kernel (%d)" code)

let ensure_compiled (spec : Program_spec.t) =
  match Hashtbl.find_opt kernel_cache spec.expression_key with
  | Some k -> k
  | None ->
      let so_file = build_shared_library ~source:spec.src in
      let handle = Dl.dlopen ~filename:so_file ~flags:[ Dl.RTLD_NOW ] in
      let compiled = { handle; function_name = spec.function_name; n_inputs = spec.n_inputs } in
      Hashtbl.replace kernel_cache spec.expression_key compiled;
      compiled

let call_kernel (k : compiled_kernel) ~(out_ptr : float ptr) ~(input_ptrs : float ptr list) ~(n : int) =
  let f name typ = foreign ~from:k.handle name typ in
  match (k.n_inputs, input_ptrs) with
  | 0, [] ->
      let fn = f k.function_name (ptr float @-> int @-> returning void) in
      fn out_ptr n
  | 1, [ in0 ] ->
      let fn = f k.function_name (ptr float @-> ptr float @-> int @-> returning void) in
      fn out_ptr in0 n
  | 2, [ in0; in1 ] ->
      let fn = f k.function_name (ptr float @-> ptr float @-> ptr float @-> int @-> returning void) in
      fn out_ptr in0 in1 n
  | 3, [ in0; in1; in2 ] ->
      let fn = f k.function_name
          (ptr float @-> ptr float @-> ptr float @-> ptr float @-> int @-> returning void)
      in
      fn out_ptr in0 in1 in2 n
  | 4, [ in0; in1; in2; in3 ] ->
      let fn = f k.function_name
          (ptr float @-> ptr float @-> ptr float @-> ptr float @-> ptr float @-> int @-> returning void)
      in
      fn out_ptr in0 in1 in2 in3 n
  | 5, [ in0; in1; in2; in3; in4 ] ->
      let fn = f k.function_name
          (ptr float @-> ptr float @-> ptr float @-> ptr float @-> ptr float @-> ptr float @-> int @-> returning void)
      in
      fn out_ptr in0 in1 in2 in3 in4 n
  | 6, [ in0; in1; in2; in3; in4; in5 ] ->
      let fn = f k.function_name
          (ptr float @-> ptr float @-> ptr float @-> ptr float @-> ptr float @-> ptr float @-> ptr float @-> int @-> returning void)
      in
      fn out_ptr in0 in1 in2 in3 in4 in5 n
  | _ ->
      failwith
        (Printf.sprintf "unsupported input arity for CPU kernel: expected %d got %d"
           k.n_inputs (List.length input_ptrs))

let available () =
  try
    let spec = C_renderer.render_expr_kernel ~expr:(Uop.Binop (Uop.Add, Uop.Input 0, Uop.Input 1)) ~ninputs:2 ~length:1 in
    ignore (ensure_compiled spec);
    Ok ()
  with exn -> Error (Printexc.to_string exn)

let validate_inputs ~shape (inputs : Buffer.t list) =
  List.for_all
    (fun b ->
      Array.length b.Buffer.shape = Array.length shape
      && Array.for_all2 ( = ) b.Buffer.shape shape)
    inputs

let run_expr ~(expr : Uop.expr) ~(inputs : Buffer.t list) ~(shape : int array) =
  if not (validate_inputs ~shape inputs) then
    Error "input shape mismatch in cpu run_expr"
  else
    try
      let n = Buffer.numel shape in
      let spec = C_renderer.render_expr_kernel ~expr ~ninputs:(List.length inputs) ~length:n in
      let compiled = ensure_compiled spec in
      let out = Buffer.create shape in
      let out_ptr = Ctypes.bigarray_start Ctypes.array1 out.Buffer.data in
      let input_ptrs =
        List.map (fun b -> Ctypes.bigarray_start Ctypes.array1 b.Buffer.data) inputs
      in
      call_kernel compiled ~out_ptr ~input_ptrs ~n;
      Ok out
    with exn -> Error (Printexc.to_string exn)

let run_binop ~(op : Uop.binop) ~(a : Buffer.t) ~(b : Buffer.t) =
  if not (Buffer.same_shape a b) then
    Error
      (Printf.sprintf "shape mismatch %s vs %s" (Buffer.pp_shape a.Buffer.shape) (Buffer.pp_shape b.Buffer.shape))
  else
    run_expr ~expr:(Uop.Binop (op, Uop.Input 0, Uop.Input 1)) ~inputs:[ a; b ] ~shape:a.Buffer.shape

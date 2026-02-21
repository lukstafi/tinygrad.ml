module Cu = Cuda

let device_name = "cuda"

let context : Cu.Context.t option ref = ref None

let ensure_context () =
  match !context with
  | Some ctx ->
      Cu.Context.set_current ctx;
      ctx
  | None ->
      Cu.init ();
      let device_count = Cu.Device.get_count () in
      if device_count <= 0 then failwith "no CUDA devices found";
      let dev = Cu.Device.get ~ordinal:0 in
      let ctx = Cu.Context.create [] dev in
      Cu.Context.set_current ctx;
      context := Some ctx;
      ctx

let available () =
  try
    ignore (ensure_context ());
    Ok ()
  with exn -> Error (Printexc.to_string exn)

type compiled_kernel = {
  func : Cu.Function.t;
  module_ : Cu.Module.t;
}

let kernel_cache : (string, compiled_kernel) Hashtbl.t = Hashtbl.create 64

let compile_expr ~(expr : Uop.expr) ~(ninputs : int) ~(length : int) =
  let spec = Cuda_renderer.render_expr_kernel ~expr ~ninputs ~length in
  match Hashtbl.find_opt kernel_cache spec.expression_key with
  | Some compiled -> (spec, compiled)
  | None ->
      let ptx =
        Nvrtc.compile_to_ptx ~cu_src:spec.src ~name:(spec.function_name ^ ".cu")
          ~options:[ "--use_fast_math" ] ~with_debug:false
      in
      let module_ = Cu.Module.load_data_ex ptx [] in
      let func = Cu.Module.get_function module_ ~name:spec.function_name in
      let compiled = { func; module_ } in
      Hashtbl.replace kernel_cache spec.expression_key compiled;
      (spec, compiled)

let validate_inputs ~shape (inputs : Buffer.t list) =
  List.for_all
    (fun (b : Buffer.t) ->
      Array.length b.shape = Array.length shape
      && Array.for_all2 ( = ) b.shape shape)
    inputs

let run_expr ~(expr : Uop.expr) ~(inputs : Buffer.t list) ~(shape : int array) =
  if not (validate_inputs ~shape inputs) then
    Error "input shape mismatch in cuda run_expr"
  else
    try
      ignore (ensure_context ());
      let n = Buffer.numel shape in
      let out = Buffer.create shape in
      if n = 0 then Ok out
      else
        let spec, compiled = compile_expr ~expr ~ninputs:(List.length inputs) ~length:n in
        ignore spec;
        let input_devs =
          List.map
            (fun (b : Buffer.t) ->
              let host = Bigarray.genarray_of_array1 b.data in
              Cu.Deviceptr.alloc_and_memcpy host)
            inputs
        in
        let out_dev = Cu.Deviceptr.mem_alloc ~size_in_bytes:(n * 4) in
        Fun.protect
          ~finally:(fun () ->
            Cu.Deviceptr.mem_free out_dev;
            List.iter Cu.Deviceptr.mem_free input_devs)
          (fun () ->
            let block = 256 in
            let grid = max 1 ((n + block - 1) / block) in
            let args =
              Cu.Stream.Tensor out_dev
              :: (List.map (fun d -> Cu.Stream.Tensor d) input_devs)
              @ [ Cu.Stream.Int n ]
            in
            Cu.Stream.launch_kernel compiled.func ~grid_dim_x:grid ~block_dim_x:block
              ~shared_mem_bytes:0 Cu.Stream.no_stream args;
            Cu.Context.synchronize ();
            let out_host = Bigarray.genarray_of_array1 out.data in
            Cu.Deviceptr.memcpy_D_to_H ~dst:out_host ~src:out_dev ();
            Ok out)
    with exn -> Error (Printexc.to_string exn)

let run_binop ~(op : Uop.binop) ~(a : Buffer.t) ~(b : Buffer.t) =
  if not (Buffer.same_shape a b) then
    Error
      (Printf.sprintf "shape mismatch %s vs %s" (Buffer.pp_shape a.shape) (Buffer.pp_shape b.shape))
  else
    run_expr ~expr:(Uop.Binop (op, Uop.Input 0, Uop.Input 1)) ~inputs:[ a; b ] ~shape:a.shape

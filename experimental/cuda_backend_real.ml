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

let compile_kernel ~(op : Uop.binop) ~(n : int) =
  let spec = Cuda_renderer.render ~op ~length:n in
  let ptx =
    Nvrtc.compile_to_ptx ~cu_src:spec.src ~name:(spec.function_name ^ ".cu")
      ~options:[ "--use_fast_math" ] ~with_debug:false
  in
  let module_ = Cu.Module.load_data_ex ptx [] in
  let func = Cu.Module.get_function module_ ~name:spec.function_name in
  spec, module_, func

let run_binop ~(op : Uop.binop) ~(a : Buffer.t) ~(b : Buffer.t) =
  if not (Buffer.same_shape a b) then
    Error
      (Printf.sprintf "shape mismatch %s vs %s" (Buffer.pp_shape a.shape) (Buffer.pp_shape b.shape))
  else
    try
      ignore (ensure_context ());
      let n = Bigarray.Array1.dim a.data in
      let _, _, func = compile_kernel ~op ~n in
      let a_host = Bigarray.genarray_of_array1 a.data in
      let b_host = Bigarray.genarray_of_array1 b.data in
      let d_a = Cu.Deviceptr.alloc_and_memcpy a_host in
      let d_b = Cu.Deviceptr.alloc_and_memcpy b_host in
      let d_out = Cu.Deviceptr.mem_alloc ~size_in_bytes:(n * 4) in
      Fun.protect
        ~finally:(fun () ->
          Cu.Deviceptr.mem_free d_out;
          Cu.Deviceptr.mem_free d_b;
          Cu.Deviceptr.mem_free d_a)
        (fun () ->
          let block = 256 in
          let grid = max 1 ((n + block - 1) / block) in
          Cu.Stream.launch_kernel func ~grid_dim_x:grid ~block_dim_x:block ~shared_mem_bytes:0
            Cu.Stream.no_stream
            [ Cu.Stream.Tensor d_out; Cu.Stream.Tensor d_a; Cu.Stream.Tensor d_b; Cu.Stream.Int n ];
          Cu.Context.synchronize ();
          let out = Buffer.create a.shape in
          let out_host = Bigarray.genarray_of_array1 out.data in
          Cu.Deviceptr.memcpy_D_to_H ~dst:out_host ~src:d_out ();
          Ok out)
    with exn -> Error (Printexc.to_string exn)

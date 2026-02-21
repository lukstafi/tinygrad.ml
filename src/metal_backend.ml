open Ctypes

let device_name = "metal"

let memcpy =
  Foreign.foreign "memcpy"
    (ptr void @-> ptr void @-> size_t @-> returning (ptr void))

let state : (Metal.Device.t * Metal.CommandQueue.t) option ref = ref None

let ensure_state () =
  match !state with
  | Some s -> s
  | None ->
      let dev = Metal.Device.create_system_default () in
      let queue = Metal.CommandQueue.on_device dev in
      state := Some (dev, queue);
      (dev, queue)

let available () =
  try
    ignore (ensure_state ());
    Ok ()
  with exn -> Error (Printexc.to_string exn)

let copy_array1_to_metal_buffer arr (buffer : Metal.Buffer.t) =
  let bytes = Bigarray.Array1.dim arr * 4 in
  let src = Ctypes.bigarray_start Ctypes.array1 arr in
  let dst = Metal.Buffer.contents buffer in
  ignore (memcpy (Ctypes.to_voidp dst) (Ctypes.to_voidp src) (Unsigned.Size_t.of_int bytes))

let copy_metal_buffer_to_array1 (buffer : Metal.Buffer.t) arr =
  let bytes = Bigarray.Array1.dim arr * 4 in
  let src = Metal.Buffer.contents buffer in
  let dst = Ctypes.bigarray_start Ctypes.array1 arr in
  ignore (memcpy (Ctypes.to_voidp dst) (Ctypes.to_voidp src) (Unsigned.Size_t.of_int bytes))

type compiled_kernel = {
  function_name : string;
  pipeline : Metal.ComputePipelineState.t;
}

let kernel_cache : (string, compiled_kernel) Hashtbl.t = Hashtbl.create 64

let compile_spec ~(dev : Metal.Device.t) (spec : Program_spec.t) =
  match Hashtbl.find_opt kernel_cache spec.expression_key with
  | Some compiled -> (spec, compiled)
  | None ->
      let opts = Metal.CompileOptions.init () in
      let lib = Metal.Library.on_device dev ~source:spec.src opts in
      let fn = Metal.Library.new_function_with_name lib spec.function_name in
      let pipeline, _ = Metal.ComputePipelineState.on_device_with_function dev fn in
      let compiled = { function_name = spec.function_name; pipeline } in
      Hashtbl.replace kernel_cache spec.expression_key compiled;
      (spec, compiled)

let compile_expr ~(dev : Metal.Device.t) ~(expr : Uop.expr) ~(ninputs : int) ~(length : int) =
  let spec = Metal_renderer.render_expr_kernel ~expr ~ninputs ~length in
  compile_spec ~dev spec

let compile_reduce ~(dev : Metal.Device.t) ~(op : Uop.reduce_op) ~(expr : Uop.expr)
    ~(ninputs : int) ~(length : int) =
  let spec = Metal_renderer.render_reduce_kernel ~op ~expr ~ninputs ~length in
  compile_spec ~dev spec

let validate_inputs ~shape (inputs : Buffer.t list) =
  List.for_all
    (fun (b : Buffer.t) ->
      Array.length b.shape = Array.length shape
      && Array.for_all2 ( = ) b.shape shape)
    inputs

let run_expr ~(expr : Uop.expr) ~(inputs : Buffer.t list) ~(shape : int array) =
  if not (validate_inputs ~shape inputs) then
    Error "input shape mismatch in metal run_expr"
  else
    try
      let n = Buffer.numel shape in
      let out = Buffer.create shape in
      if n = 0 then Ok out
      else
        let dev, queue = ensure_state () in
        let spec, compiled = compile_expr ~dev ~expr ~ninputs:(List.length inputs) ~length:n in
        ignore spec;
        let bytes = n * 4 in
        let ro = Metal.ResourceOptions.storage_mode_shared in
        let out_buf = Metal.Buffer.on_device dev ~length:bytes ro in
        let in_bufs =
          List.map
            (fun (b : Buffer.t) ->
              let mbuf = Metal.Buffer.on_device dev ~length:bytes ro in
              copy_array1_to_metal_buffer b.data mbuf;
              mbuf)
            inputs
        in
        let cmd = Metal.CommandBuffer.on_queue queue in
        let enc = Metal.ComputeCommandEncoder.on_buffer cmd in
        Metal.ComputeCommandEncoder.set_compute_pipeline_state enc compiled.pipeline;
        Metal.ComputeCommandEncoder.set_buffer enc ~index:0 out_buf;
        List.iteri (fun i b -> Metal.ComputeCommandEncoder.set_buffer enc ~index:(i + 1) b) in_bufs;
        let n_ptr = Ctypes.allocate Ctypes.int n in
        Metal.ComputeCommandEncoder.set_bytes enc ~bytes:(Ctypes.to_voidp n_ptr)
          ~length:(Ctypes.sizeof Ctypes.int) ~index:(List.length inputs + 1);
        let width =
          let max_width = Metal.ComputePipelineState.get_max_total_threads_per_threadgroup compiled.pipeline in
          max 1 (min n (min 256 max_width))
        in
        let groups = (n + width - 1) / width in
        Metal.ComputeCommandEncoder.dispatch_threadgroups enc
          ~threadgroups_per_grid:{ Metal.Size.width = groups; height = 1; depth = 1 }
          ~threads_per_threadgroup:{ Metal.Size.width = width; height = 1; depth = 1 };
        Metal.ComputeCommandEncoder.end_encoding enc;
        Metal.CommandBuffer.commit cmd;
        Metal.CommandBuffer.wait_until_completed cmd;
        begin
          match Metal.CommandBuffer.get_error cmd with
          | None -> ()
          | Some err -> failwith err
        end;
        copy_metal_buffer_to_array1 out_buf out.data;
        Ok out
    with exn -> Error (Printexc.to_string exn)

let run_reduce ~(op : Uop.reduce_op) ~(expr : Uop.expr) ~(inputs : Buffer.t list)
    ~(shape : int array) =
  if not (validate_inputs ~shape inputs) then
    Error "input shape mismatch in metal run_reduce"
  else
    try
      let n = Buffer.numel shape in
      if n = 0 then
        Ok (match op with Uop.Sum -> 0.0 | Uop.Max -> Float.neg_infinity)
      else
        let dev, queue = ensure_state () in
        let spec, compiled = compile_reduce ~dev ~op ~expr ~ninputs:(List.length inputs) ~length:n in
        ignore spec;
        let ro = Metal.ResourceOptions.storage_mode_shared in
        let out_buf = Metal.Buffer.on_device dev ~length:4 ro in
        let in_bufs =
          List.map
            (fun (b : Buffer.t) ->
              let bytes = Buffer.numel b.shape * 4 in
              let mbuf = Metal.Buffer.on_device dev ~length:bytes ro in
              copy_array1_to_metal_buffer b.data mbuf;
              mbuf)
            inputs
        in
        let cmd = Metal.CommandBuffer.on_queue queue in
        let enc = Metal.ComputeCommandEncoder.on_buffer cmd in
        Metal.ComputeCommandEncoder.set_compute_pipeline_state enc compiled.pipeline;
        Metal.ComputeCommandEncoder.set_buffer enc ~index:0 out_buf;
        List.iteri (fun i b -> Metal.ComputeCommandEncoder.set_buffer enc ~index:(i + 1) b) in_bufs;
        let n_ptr = Ctypes.allocate Ctypes.int n in
        Metal.ComputeCommandEncoder.set_bytes enc ~bytes:(Ctypes.to_voidp n_ptr)
          ~length:(Ctypes.sizeof Ctypes.int) ~index:(List.length inputs + 1);
        Metal.ComputeCommandEncoder.dispatch_threadgroups enc
          ~threadgroups_per_grid:{ Metal.Size.width = 1; height = 1; depth = 1 }
          ~threads_per_threadgroup:{ Metal.Size.width = 1; height = 1; depth = 1 };
        Metal.ComputeCommandEncoder.end_encoding enc;
        Metal.CommandBuffer.commit cmd;
        Metal.CommandBuffer.wait_until_completed cmd;
        begin
          match Metal.CommandBuffer.get_error cmd with
          | None -> ()
          | Some err -> failwith err
        end;
        let out = Buffer.create [| 1 |] in
        copy_metal_buffer_to_array1 out_buf out.data;
        Ok out.data.{0}
    with exn -> Error (Printexc.to_string exn)

let run_binop ~(op : Uop.binop) ~(a : Buffer.t) ~(b : Buffer.t) =
  if not (Buffer.same_shape a b) then
    Error
      (Printf.sprintf "shape mismatch %s vs %s" (Buffer.pp_shape a.shape) (Buffer.pp_shape b.shape))
  else
    run_expr ~expr:(Uop.Binop (op, Uop.Input 0, Uop.Input 1)) ~inputs:[ a; b ] ~shape:a.shape

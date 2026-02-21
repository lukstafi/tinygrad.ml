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

let copy_array1_to_buffer arr buffer =
  let bytes = Bigarray.Array1.dim arr * 4 in
  let src = Ctypes.bigarray_start Ctypes.array1 arr in
  let dst = Metal.Buffer.contents buffer in
  ignore (memcpy (Ctypes.to_voidp dst) (Ctypes.to_voidp src) (Unsigned.Size_t.of_int bytes))

let copy_buffer_to_array1 buffer arr =
  let bytes = Bigarray.Array1.dim arr * 4 in
  let src = Metal.Buffer.contents buffer in
  let dst = Ctypes.bigarray_start Ctypes.array1 arr in
  ignore (memcpy (Ctypes.to_voidp dst) (Ctypes.to_voidp src) (Unsigned.Size_t.of_int bytes))

let run_binop ~(op : Uop.binop) ~(a : Buffer.t) ~(b : Buffer.t) =
  if not (Buffer.same_shape a b) then
    Error
      (Printf.sprintf "shape mismatch %s vs %s" (Buffer.pp_shape a.shape) (Buffer.pp_shape b.shape))
  else
    try
      let n = Bigarray.Array1.dim a.data in
      let out = Buffer.create a.shape in
      if n = 0 then Ok out
      else
        let dev, queue = ensure_state () in
        let spec = Metal_renderer.render ~op ~length:n in
        let opts = Metal.CompileOptions.init () in
        let lib = Metal.Library.on_device dev ~source:spec.src opts in
        let fn = Metal.Library.new_function_with_name lib spec.function_name in
        let pipeline, _ = Metal.ComputePipelineState.on_device_with_function dev fn in
        let bytes = n * 4 in
        let ro = Metal.ResourceOptions.storage_mode_shared in
        let out_buf = Metal.Buffer.on_device dev ~length:bytes ro in
        let a_buf = Metal.Buffer.on_device dev ~length:bytes ro in
        let b_buf = Metal.Buffer.on_device dev ~length:bytes ro in
        copy_array1_to_buffer a.data a_buf;
        copy_array1_to_buffer b.data b_buf;
        let cmd = Metal.CommandBuffer.on_queue queue in
        let enc = Metal.ComputeCommandEncoder.on_buffer cmd in
        Metal.ComputeCommandEncoder.set_compute_pipeline_state enc pipeline;
        Metal.ComputeCommandEncoder.set_buffer enc ~index:0 out_buf;
        Metal.ComputeCommandEncoder.set_buffer enc ~index:1 a_buf;
        Metal.ComputeCommandEncoder.set_buffer enc ~index:2 b_buf;
        let n_ptr = Ctypes.allocate Ctypes.int n in
        Metal.ComputeCommandEncoder.set_bytes enc ~bytes:(Ctypes.to_voidp n_ptr)
          ~length:(Ctypes.sizeof Ctypes.int) ~index:3;
        let width =
          let max_width = Metal.ComputePipelineState.get_max_total_threads_per_threadgroup pipeline in
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
        copy_buffer_to_array1 out_buf out.data;
        Ok out
    with exn -> Error (Printexc.to_string exn)

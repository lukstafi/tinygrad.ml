(** Metal GPU backend for tinygrad_ml.
    Uses the OCaml `metal` package to compile MSL kernels and execute them.
    Ported from tinygrad/runtime/ops_metal.py (simplified). *)

open Ctypes

(** Lazy-initialized Metal device and command queue *)
let device_ref : Metal.Device.t option ref = ref None
let queue_ref : Metal.CommandQueue.t option ref = ref None

let get_device () =
  match !device_ref with
  | Some d -> d
  | None ->
    let d = Metal.Device.create_system_default () in
    device_ref := Some d;
    d

let get_queue () =
  match !queue_ref with
  | Some q -> q
  | None ->
    let dev = get_device () in
    let q = Metal.CommandQueue.on_device dev in
    queue_ref := Some q;
    q

(** Side table: nativeint address -> Metal.Buffer.t, to keep buffers alive and
    retrieve them for kernel dispatch *)
let metal_buffers : (nativeint, Metal.Buffer.t) Hashtbl.t = Hashtbl.create 64

(** Allocate a Metal buffer of [nbytes] bytes, returns nativeint handle.
    We use shared storage so CPU and GPU can both access the memory. *)
let alloc nbytes =
  let dev = get_device () in
  let buf = Metal.Buffer.on_device dev ~length:nbytes
    Metal.ResourceOptions.storage_mode_shared in
  (* Store the Metal.Buffer.t in a side table keyed by the contents pointer *)
  let ptr = Metal.Buffer.contents buf in
  let addr = raw_address_of_ptr (to_voidp ptr) in
  (* We need to keep the Metal buffer alive — store it *)
  Hashtbl.replace metal_buffers addr buf;
  addr

let free ptr =
  Hashtbl.remove metal_buffers ptr

(** Copy host data into a Metal shared buffer *)
let copyin dst_ptr src =
  let n = Bigarray.Array1.dim src in
  let dst_p = coerce (ptr void) (ptr char) (ptr_of_raw_address dst_ptr) in
  let dst_ba = bigarray_of_ptr array1 n Bigarray.char dst_p in
  Bigarray.Array1.blit src dst_ba

(** Copy Metal shared buffer data to host *)
let copyout dst src_ptr =
  let n = Bigarray.Array1.dim dst in
  let src_p = coerce (ptr void) (ptr char) (ptr_of_raw_address src_ptr) in
  let src_ba = bigarray_of_ptr array1 n Bigarray.char src_p in
  Bigarray.Array1.blit src_ba dst

(** Kernel cache: (kernel_name, source) -> (pipeline, function_name) *)
let kernel_cache : (string, Metal.ComputePipelineState.t) Hashtbl.t = Hashtbl.create 16

(** Compile MSL source code, returns the source string as "binary" identifier.
    The actual compilation happens at exec time via the Metal library API. *)
let compile _name src =
  (* For Metal, we store the source and compile at exec time.
     Return the source as the "path" — we'll use it as a cache key. *)
  src

(** Execute a compiled Metal kernel *)
let exec kernel_name src buf_ptrs _int_vals =
  let dev = get_device () in
  let queue = get_queue () in

  (* Get or compile the pipeline *)
  let cache_key = kernel_name ^ ":" ^ src in
  let pipeline = match Hashtbl.find_opt kernel_cache cache_key with
    | Some p -> p
    | None ->
      let opts = Metal.CompileOptions.init () in
      let lib = Metal.Library.on_device dev ~source:src opts in
      let func = Metal.Library.new_function_with_name lib kernel_name in
      let pso, _ = Metal.ComputePipelineState.on_device_with_function dev func in
      Hashtbl.replace kernel_cache cache_key pso;
      pso
  in

  (* Create command buffer and encoder *)
  let cmd = Metal.CommandBuffer.on_queue queue in
  let enc = Metal.ComputeCommandEncoder.on_buffer cmd in
  Metal.ComputeCommandEncoder.set_compute_pipeline_state enc pipeline;

  (* Bind buffers *)
  List.iteri (fun i ptr ->
    match Hashtbl.find_opt metal_buffers ptr with
    | Some metal_buf ->
      Metal.ComputeCommandEncoder.set_buffer enc ~index:i metal_buf
    | None ->
      failwith (Printf.sprintf "Metal.exec: buffer at index %d not found in metal_buffers table" i)
  ) buf_ptrs;

  (* Dispatch — use a simple 1D grid *)
  let max_threads = Metal.ComputePipelineState.get_max_total_threads_per_threadgroup pipeline in
  let width = min 256 max_threads in
  (* We need to figure out the grid size from the kernel.
     For now, use a reasonable default based on buffer sizes.
     The kernel itself bounds-checks via its loop. *)
  let groups = 1 in  (* For loop-based kernels, 1 group is enough *)
  Metal.ComputeCommandEncoder.dispatch_threadgroups enc
    ~threadgroups_per_grid:{ Metal.Size.width = groups; height = 1; depth = 1 }
    ~threads_per_threadgroup:{ Metal.Size.width = width; height = 1; depth = 1 };
  Metal.ComputeCommandEncoder.end_encoding enc;
  Metal.CommandBuffer.commit cmd;
  Metal.CommandBuffer.wait_until_completed cmd;

  (* Check for errors *)
  (match Metal.CommandBuffer.get_error cmd with
   | None -> ()
   | Some err -> failwith (Printf.sprintf "Metal kernel execution failed: %s" err))

let synchronize () =
  (* Metal command buffers are synchronous with wait_until_completed *)
  ()

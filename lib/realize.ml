(** Realize: compile and execute scheduled kernels.
    Ported from tinygrad/engine/realize.py (simplified).

    This module takes exec_items from the scheduler and actually executes them:
    - Copyin items copy float data into device buffers
    - Kernel items compile and execute UOp kernels *)

let run_schedule (schedule : Schedule.exec_item list) =
  List.iter (fun item ->
    match item with
    | Schedule.Copyin { buf; data } ->
      Device.copyin_floats buf data
    | Schedule.Kernel { name; uops; bufs; data_map } ->
      (* Copy input data into buffers *)
      List.iter (fun (idx, data) ->
        Device.copyin_floats (List.nth bufs idx) data
      ) data_map;
      (* Render the kernel *)
      let device = (List.hd bufs).device in
      let cfg = match String.uppercase_ascii device with
        | "METAL" -> Cstyle.metal_config
        | "CUDA" -> Cstyle.cuda_config ~arch:"sm_80"
        | _ -> Cstyle.clang_config
      in
      let pspec = Cstyle.render_uops cfg uops in
      (* Compile *)
      let module B = (val Device.get_backend device : Device.Backend) in
      let binary = B.compile name pspec.src in
      (* Execute *)
      let ptrs = List.map (fun (b : Device.buffer) -> b.ptr) bufs in
      B.exec name binary ptrs [];
      ignore pspec
  ) schedule

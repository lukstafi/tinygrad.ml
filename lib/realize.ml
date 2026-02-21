(** Realize: compile and execute scheduled kernels.
    Ported from tinygrad/engine/realize.py (simplified).

    This module takes exec_items from the scheduler and actually executes them:
    - Copyin items copy float data into device buffers
    - Kernel items render, compile, and execute UOp kernel graphs *)

let run_schedule (schedule : Schedule.exec_item list) =
  List.iter (fun item ->
    match item with
    | Schedule.Copyin { buf; data } ->
      Device.copyin_floats buf data
    | Schedule.Kernel { name; uops; bufs } ->
      let device = (List.hd bufs).device in
      let cfg = match String.uppercase_ascii device with
        | "METAL" -> Cstyle.metal_config
        | "CUDA" -> Cstyle.cuda_config ~arch:"sm_80"
        | _ -> Cstyle.clang_config
      in
      let pspec = Cstyle.render_uops cfg uops in
      let module B = (val Device.get_backend device : Device.Backend) in
      let binary = B.compile name pspec.src in
      (* Reorder buffers according to pspec.globals (rendered param order) *)
      let ordered_ptrs = List.map (fun idx ->
        (List.nth bufs idx).Device.ptr
      ) pspec.globals in
      B.exec name binary ordered_ptrs []
  ) schedule

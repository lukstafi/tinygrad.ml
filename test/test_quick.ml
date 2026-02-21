(* Quick test to verify all modules work *)
let () =
  Printf.printf "Starting quick tests...\n%!";

  (* Helpers *)
  assert (Helpers.prod [2;3] = 6);
  assert (Dtype.bitsize Dtype.float32 = 32);
  assert (Ops.to_string Ops.ADD = "ADD");
  Printf.printf "  Basic modules ok\n%!";

  (* UOp *)
  let c1 = Uop.const Dtype.float32 1.0 in
  let c2 = Uop.const Dtype.float32 2.0 in
  let sum = Uop.add c1 c2 in
  assert (List.length (Uop.toposort1 sum) = 3);
  assert (Uop.equal c1 (Uop.const Dtype.float32 1.0));
  Printf.printf "  UOp ok\n%!";

  (* Pattern matcher *)
  let fold_add : unit Pattern_matcher.rule = {
    pattern = Pattern_matcher.pat ~ops:[Ops.ADD]
      ~src:[
        Pattern_matcher.pat ~ops:[Ops.CONST] ~name:"a" ();
        Pattern_matcher.pat ~ops:[Ops.CONST] ~name:"b" ();
      ] ();
    rewrite = (fun () bindings ->
      let a = Pattern_matcher.find "a" bindings in
      let b = Pattern_matcher.find "b" bindings in
      match a.arg, b.arg with
      | Uop.Float_arg fa, Uop.Float_arg fb ->
        Some (Uop.const_float a.dtype (fa +. fb))
      | _ -> None
    );
  } in
  let pm = Pattern_matcher.create [fold_add] in
  let result = Pattern_matcher.graph_rewrite pm () sum in
  assert (result.Uop.op = Ops.CONST);
  let c3 = Uop.const Dtype.float32 3.0 in
  let nested = Uop.add sum c3 in
  let result2 = Pattern_matcher.graph_rewrite pm () nested in
  assert (result2.op = Ops.CONST);
  Printf.printf "  Pattern matcher ok\n%!";

  (* Build a kernel: out[i] = in[i] + 1.0 *)
  let in_param = Uop.param 0 (Dtype.ptr Dtype.float32) in
  let out_param = Uop.param 1 (Dtype.ptr Dtype.float32) in
  let i = Uop.range (Uop.const_int Dtype.int32 1024) [0; 0] in
  let in_idx = Uop.index in_param i in
  let val_ = Uop.load in_idx in
  let one = Uop.const Dtype.float32 1.0 in
  let result_val = Uop.add val_ one in
  let out_idx = Uop.index out_param i in
  let st = Uop.store out_idx result_val in
  let end_range = Uop.end_ i in
  let kernel = Uop.sink ~name:"add_one" [st; end_range] in
  let uops = Uop.toposort1 kernel in

  (* CPU C *)
  let pspec = Cstyle.render_uops Cstyle.clang_config uops in
  Printf.printf "\n=== CPU C ===\n%s\n%!" pspec.src;

  (* CUDA *)
  let cuda_pspec = Cstyle.render_uops (Cstyle.cuda_config ~arch:"sm_80") uops in
  Printf.printf "\n=== CUDA ===\n%s\n%!" cuda_pspec.src;

  (* Metal *)
  let metal_pspec = Cstyle.render_uops Cstyle.metal_config uops in
  Printf.printf "\n=== Metal ===\n%s\n%!" metal_pspec.src;

  (* Test Tensor *)
  let t1 = Tensor.zeros [3; 4] in
  let t2 = Tensor.ones [3; 4] in
  let t3 = Tensor.add t1 t2 in
  assert (t3.shape = [3; 4]);
  let t4 = Tensor.reshape t1 [12] in
  assert (t4.shape = [12]);
  let t5 = Tensor.sum ~axes:[1] t1 in
  assert (t5.shape = [3; 1]);
  Printf.printf "  Tensor ok\n%!";

  (* Test Gradient *)
  let x = Uop.const Dtype.float32 3.0 in
  let y = Uop.const Dtype.float32 2.0 in
  let z = Uop.mul x y in  (* z = x*y, dz/dx = y, dz/dy = x *)
  let grad_z = Uop.const Dtype.float32 1.0 in
  let grads = Gradient.compute_gradient z grad_z [x; y] in
  assert (List.length grads = 2);
  Printf.printf "  Gradient ok\n%!";

  (* Test CPU compilation *)
  Printf.printf "\n  Testing CPU C compilation...\n%!";
  (try
    let module B = (val Device.get_backend "CPU" : Device.Backend) in
    let so_path = B.compile "add_one" pspec.src in
    Printf.printf "  Compiled to %s\n%!" so_path;
    Printf.printf "  CPU compilation ok\n%!";

    (* End-to-end CPU execution test: out[i] = in[i] + 1.0 *)
    Printf.printf "\n  Testing end-to-end CPU execution...\n%!";
    let n = 1024 in
    let in_buf = Device.make_buffer ~device:"CPU" ~size:n ~dtype:Dtype.float32 in
    let out_buf = Device.make_buffer ~device:"CPU" ~size:n ~dtype:Dtype.float32 in
    let _ = Device.alloc_buffer in_buf in
    let _ = Device.alloc_buffer out_buf in

    (* Fill input with 0.0, 1.0, 2.0, ... *)
    let in_data = Array.init n (fun i -> Float.of_int i) in
    Device.copyin_floats in_buf in_data;

    (* Execute kernel: map param indices to buffers via pspec.globals *)
    let bufs_by_param = [| in_buf; out_buf |] in  (* param 0 = in, param 1 = out *)
    let ordered_ptrs = List.map (fun idx -> bufs_by_param.(idx).ptr) pspec.globals in
    B.exec "add_one" so_path ordered_ptrs [];

    (* Read back output *)
    let out_data = Device.copyout_floats out_buf in

    (* Verify: out[i] should be i + 1.0 *)
    let correct = ref true in
    for i = 0 to n - 1 do
      let expected = Float.of_int i +. 1.0 in
      if Float.abs (out_data.(i) -. expected) > 1e-6 then begin
        Printf.printf "  MISMATCH at %d: got %f, expected %f\n%!" i out_data.(i) expected;
        correct := false
      end
    done;
    if !correct then
      Printf.printf "  END-TO-END CPU EXECUTION PASSED! out[0]=%f out[1]=%f out[1023]=%f\n%!"
        out_data.(0) out_data.(1) out_data.(1023)
    else
      Printf.printf "  END-TO-END CPU EXECUTION FAILED\n%!";

  with e ->
    Printf.printf "  CPU test failed: %s\n%!" (Printexc.to_string e));

  Printf.printf "\nAll quick tests passed!\n%!"

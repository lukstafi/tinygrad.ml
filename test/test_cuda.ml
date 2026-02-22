open Test_helpers

let test_matches_cpu () =
  let a = Tinygrad_ml.Tensor.from_array [| 2.0; 4.0; 6.0; 8.0 |] in
  let b = Tinygrad_ml.Tensor.from_array [| 1.5; 2.5; 3.5; 4.5 |] in
  let expr =
    Tinygrad_ml.Tensor.sqrt
      (Tinygrad_ml.Tensor.mul (Tinygrad_ml.Tensor.add a b) (Tinygrad_ml.Tensor.sub a b))
  in
  let cpu = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c expr in
  let cuda = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cuda expr in
  check_array ~msg:"cuda matches cpu for fused expr" cpu cuda

let test_chain_cuda () =
  let a = Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0; 4.0 |] in
  let b = Tinygrad_ml.Tensor.from_array [| 10.0; 20.0; 30.0; 40.0 |] in
  let c = Tinygrad_ml.Tensor.add a b in
  let c_realized = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cuda c in
  check_array ~msg:"cuda chain c=a+b" [| 11.0; 22.0; 33.0; 44.0 |] c_realized;
  let d = Tinygrad_ml.Tensor.mul c a in
  let d_realized = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cuda d in
  check_array ~msg:"cuda chain d=(a+b)*a" [| 11.0; 44.0; 99.0; 176.0 |] d_realized

let test_reductions_cuda () =
  let x = Tinygrad_ml.Tensor.from_array [| 2.0; 7.0; 5.0; 4.0 |] in
  let s = Tinygrad_ml.Tensor.sum ~device:Tinygrad_ml.Runtime.Cuda x in
  check_close ~msg:"cuda sum" 18.0 s;
  let m = Tinygrad_ml.Tensor.max ~device:Tinygrad_ml.Runtime.Cuda x in
  check_close ~msg:"cuda max" 7.0 m;
  let mean = Tinygrad_ml.Tensor.mean ~device:Tinygrad_ml.Runtime.Cuda x in
  check_close ~msg:"cuda mean" 4.5 mean

let test_fused_reductions_cuda_vs_cpu () =
  let a = Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  let b = Tinygrad_ml.Tensor.from_array [| 5.0; 4.0; 3.0; 2.0; 1.0 |] in
  let fused =
    Tinygrad_ml.Tensor.reciprocal
      (Tinygrad_ml.Tensor.sqrt (Tinygrad_ml.Tensor.mul (Tinygrad_ml.Tensor.add a b) a))
  in
  let cpu_sum = Tinygrad_ml.Tensor.sum ~device:Tinygrad_ml.Runtime.Cpu_c fused in
  let cuda_sum = Tinygrad_ml.Tensor.sum ~device:Tinygrad_ml.Runtime.Cuda fused in
  let cpu_max = Tinygrad_ml.Tensor.max ~device:Tinygrad_ml.Runtime.Cpu_c fused in
  let cuda_max = Tinygrad_ml.Tensor.max ~device:Tinygrad_ml.Runtime.Cuda fused in
  check_close ~msg:"cuda fused sum matches cpu" cpu_sum cuda_sum;
  check_close ~msg:"cuda fused max matches cpu" cpu_max cuda_max

let test_axis_reductions_cuda_vs_cpu () =
  let a =
    Tinygrad_ml.Tensor.reshape
      (Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0 |])
      [| 2; 4 |]
  in
  let cpu_sum_axis =
    Tinygrad_ml.Tensor.sum_axis ~device:Tinygrad_ml.Runtime.Cpu_c ~axes:[ 1 ] a
    |> Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c
  in
  let cuda_sum_axis =
    Tinygrad_ml.Tensor.sum_axis ~device:Tinygrad_ml.Runtime.Cuda ~axes:[ 1 ] a
    |> Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c
  in
  check_array ~msg:"cuda sum_axis matches cpu" cpu_sum_axis cuda_sum_axis;
  let b =
    Tinygrad_ml.Tensor.reshape
      (Tinygrad_ml.Tensor.from_array (Array.init 24 (fun i -> float_of_int (i + 1))))
      [| 2; 3; 4 |]
  in
  let cpu_noncontig =
    Tinygrad_ml.Tensor.sum_axis ~device:Tinygrad_ml.Runtime.Cpu_c ~axes:[ 0; 2 ] b
    |> Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c
  in
  let cuda_noncontig =
    Tinygrad_ml.Tensor.sum_axis ~device:Tinygrad_ml.Runtime.Cuda ~axes:[ 0; 2 ] b
    |> Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c
  in
  check_array ~msg:"cuda noncontig sum_axis matches cpu" cpu_noncontig cuda_noncontig

let test_movement_ops_cuda_vs_cpu () =
  let x =
    Tinygrad_ml.Tensor.reshape
      (Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |])
      [| 2; 3 |]
  in
  let flip_expr = Tinygrad_ml.Tensor.flip x [| 1 |] in
  let cpu_flip = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c flip_expr in
  let cuda_flip = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cuda flip_expr in
  check_array ~msg:"cuda flip matches cpu" cpu_flip cuda_flip;
  let pad_expr = Tinygrad_ml.Tensor.pad x [| (1, 0); (0, 1) |] in
  let cpu_pad = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c pad_expr in
  let cuda_pad = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cuda pad_expr in
  check_array ~msg:"cuda pad matches cpu" cpu_pad cuda_pad;
  let shrink_expr = Tinygrad_ml.Tensor.shrink x [| (0, 2); (1, 3) |] in
  let cpu_shrink = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c shrink_expr in
  let cuda_shrink = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cuda shrink_expr in
  check_array ~msg:"cuda shrink matches cpu" cpu_shrink cuda_shrink

let run_or_skip () =
  match Tinygrad_ml.Cuda_backend.available () with
  | Error msg ->
      Printf.printf "test_cuda: skipped (%s)\n%!" msg;
      ()
  | Ok () ->
      test_matches_cpu ();
      test_chain_cuda ();
      test_reductions_cuda ();
      test_fused_reductions_cuda_vs_cpu ();
      test_axis_reductions_cuda_vs_cpu ();
      test_movement_ops_cuda_vs_cpu ();
      Printf.printf "test_cuda: ok\n%!"

let () = run_or_skip ()

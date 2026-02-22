open Test_helpers

let test_add_mul_metal () =
  let a = Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0; 4.0 |] in
  let b = Tinygrad_ml.Tensor.from_array [| 10.0; 20.0; 30.0; 40.0 |] in
  let expr = Tinygrad_ml.Tensor.mul (Tinygrad_ml.Tensor.add a b) a in
  let out = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Metal expr in
  check_array ~msg:"metal (a+b)*a" [| 11.0; 44.0; 99.0; 176.0 |] out

let test_unary_chain_metal () =
  let x = Tinygrad_ml.Tensor.from_array [| 1.0; 4.0; 9.0; 16.0 |] in
  let expr =
    Tinygrad_ml.Tensor.reciprocal
      (Tinygrad_ml.Tensor.sqrt (Tinygrad_ml.Tensor.neg (Tinygrad_ml.Tensor.neg x)))
  in
  let out = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Metal expr in
  check_array ~msg:"metal reciprocal(sqrt(neg(neg(x))))" [| 1.0; 0.5; 1.0 /. 3.0; 0.25 |] out

let test_matches_cpu () =
  let a = Tinygrad_ml.Tensor.from_array [| 3.0; 5.0; 7.0; 9.0; 11.0 |] in
  let b = Tinygrad_ml.Tensor.from_array [| 1.5; 2.5; 3.5; 4.5; 5.5 |] in
  let expr =
    Tinygrad_ml.Tensor.sqrt
      (Tinygrad_ml.Tensor.mul (Tinygrad_ml.Tensor.add a b) (Tinygrad_ml.Tensor.sub a b))
  in
  let cpu = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c expr in
  let metal = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Metal expr in
  check_array ~msg:"metal matches cpu for fused expr" cpu metal

let test_chain_realize_metal () =
  let a = Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0; 4.0 |] in
  let b = Tinygrad_ml.Tensor.from_array [| 10.0; 20.0; 30.0; 40.0 |] in
  let c = Tinygrad_ml.Tensor.add a b in
  let c_realized = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Metal c in
  check_array ~msg:"metal chain c=a+b" [| 11.0; 22.0; 33.0; 44.0 |] c_realized;
  let d = Tinygrad_ml.Tensor.mul c a in
  let d_realized = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Metal d in
  check_array ~msg:"metal chain d=(a+b)*a" [| 11.0; 44.0; 99.0; 176.0 |] d_realized

let test_reductions_metal () =
  let x = Tinygrad_ml.Tensor.from_array [| 2.0; 7.0; 5.0; 4.0 |] in
  let s = Tinygrad_ml.Tensor.sum ~device:Tinygrad_ml.Runtime.Metal x in
  check_close ~msg:"metal sum" 18.0 s;
  let m = Tinygrad_ml.Tensor.max ~device:Tinygrad_ml.Runtime.Metal x in
  check_close ~msg:"metal max" 7.0 m;
  let mean = Tinygrad_ml.Tensor.mean ~device:Tinygrad_ml.Runtime.Metal x in
  check_close ~msg:"metal mean" 4.5 mean

let test_fused_reductions_metal_vs_cpu () =
  let a = Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0; 4.0; 5.0 |] in
  let b = Tinygrad_ml.Tensor.from_array [| 5.0; 4.0; 3.0; 2.0; 1.0 |] in
  let fused =
    Tinygrad_ml.Tensor.reciprocal
      (Tinygrad_ml.Tensor.sqrt (Tinygrad_ml.Tensor.mul (Tinygrad_ml.Tensor.add a b) a))
  in
  let cpu_sum = Tinygrad_ml.Tensor.sum ~device:Tinygrad_ml.Runtime.Cpu_c fused in
  let metal_sum = Tinygrad_ml.Tensor.sum ~device:Tinygrad_ml.Runtime.Metal fused in
  let cpu_max = Tinygrad_ml.Tensor.max ~device:Tinygrad_ml.Runtime.Cpu_c fused in
  let metal_max = Tinygrad_ml.Tensor.max ~device:Tinygrad_ml.Runtime.Metal fused in
  check_close ~msg:"metal fused sum matches cpu" cpu_sum metal_sum;
  check_close ~msg:"metal fused max matches cpu" cpu_max metal_max

let test_axis_reductions_metal_vs_cpu () =
  let a =
    Tinygrad_ml.Tensor.reshape
      (Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0 |])
      [| 2; 4 |]
  in
  let cpu_sum_axis =
    Tinygrad_ml.Tensor.sum_axis ~device:Tinygrad_ml.Runtime.Cpu_c ~axes:[ 1 ] a
    |> Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c
  in
  let metal_sum_axis =
    Tinygrad_ml.Tensor.sum_axis ~device:Tinygrad_ml.Runtime.Metal ~axes:[ 1 ] a
    |> Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c
  in
  check_array ~msg:"metal sum_axis matches cpu" cpu_sum_axis metal_sum_axis;
  let cpu_max_axis =
    Tinygrad_ml.Tensor.max_axis ~device:Tinygrad_ml.Runtime.Cpu_c ~axes:[ 0 ] a
    |> Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c
  in
  let metal_max_axis =
    Tinygrad_ml.Tensor.max_axis ~device:Tinygrad_ml.Runtime.Metal ~axes:[ 0 ] a
    |> Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c
  in
  check_array ~msg:"metal max_axis matches cpu" cpu_max_axis metal_max_axis;
  let b =
    Tinygrad_ml.Tensor.reshape
      (Tinygrad_ml.Tensor.from_array (Array.init 24 (fun i -> float_of_int (i + 1))))
      [| 2; 3; 4 |]
  in
  let cpu_noncontig =
    Tinygrad_ml.Tensor.sum_axis ~device:Tinygrad_ml.Runtime.Cpu_c ~axes:[ 0; 2 ] b
    |> Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c
  in
  let metal_noncontig =
    Tinygrad_ml.Tensor.sum_axis ~device:Tinygrad_ml.Runtime.Metal ~axes:[ 0; 2 ] b
    |> Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c
  in
  check_array ~msg:"metal noncontig sum_axis matches cpu" cpu_noncontig metal_noncontig

let test_movement_ops_metal_vs_cpu () =
  let x =
    Tinygrad_ml.Tensor.reshape
      (Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |])
      [| 2; 3 |]
  in
  let flip_expr = Tinygrad_ml.Tensor.flip x [| 1 |] in
  let cpu_flip = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c flip_expr in
  let metal_flip = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Metal flip_expr in
  check_array ~msg:"metal flip matches cpu" cpu_flip metal_flip;
  let pad_expr = Tinygrad_ml.Tensor.pad x [| (1, 0); (0, 1) |] in
  let cpu_pad = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c pad_expr in
  let metal_pad = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Metal pad_expr in
  check_array ~msg:"metal pad matches cpu" cpu_pad metal_pad;
  let shrink_expr = Tinygrad_ml.Tensor.shrink x [| (0, 2); (1, 3) |] in
  let cpu_shrink = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c shrink_expr in
  let metal_shrink = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Metal shrink_expr in
  check_array ~msg:"metal shrink matches cpu" cpu_shrink metal_shrink

let test_where_cast_metal_vs_cpu () =
  let cond = Tinygrad_ml.Tensor.from_array [| 1.0; 0.0; 1.0; 0.0 |] in
  let t = Tinygrad_ml.Tensor.from_array [| 2.0; Float.nan; 4.0; Float.nan |] in
  let f = Tinygrad_ml.Tensor.from_array [| Float.nan; 30.0; Float.nan; 50.0 |] in
  let where_expr = Tinygrad_ml.Tensor.where_ cond t f in
  let cpu_where = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c where_expr in
  let metal_where = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Metal where_expr in
  check_array ~msg:"metal where matches cpu" cpu_where metal_where;
  check_array ~msg:"metal where branch selection"
    [| 2.0; 30.0; 4.0; 50.0 |]
    metal_where;
  let x = Tinygrad_ml.Tensor.from_array [| -1.9; -0.1; 0.0; 2.7 |] in
  let cast_i32 = Tinygrad_ml.Tensor.cast ~dtype:Tinygrad_ml.Dtype.I32 x in
  let cast_bool = Tinygrad_ml.Tensor.cast ~dtype:Tinygrad_ml.Dtype.Bool x in
  let cast_f32 = Tinygrad_ml.Tensor.cast ~dtype:Tinygrad_ml.Dtype.F32 x in
  let cpu_i32 = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c cast_i32 in
  let metal_i32 = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Metal cast_i32 in
  let cpu_bool = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c cast_bool in
  let metal_bool = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Metal cast_bool in
  let cpu_f32 = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c cast_f32 in
  let metal_f32 = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Metal cast_f32 in
  check_array ~msg:"metal cast i32 matches cpu" cpu_i32 metal_i32;
  check_array ~msg:"metal cast bool matches cpu" cpu_bool metal_bool;
  check_array ~msg:"metal cast f32 matches cpu" cpu_f32 metal_f32

let test_where_backward_metal () =
  let cond = Tinygrad_ml.Tensor.from_array [| 1.0; 0.0; 1.0; 0.0 |] in
  let t = Tinygrad_ml.Tensor.from_array [| 10.0; 20.0; 30.0; 40.0 |] in
  let f = Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0; 4.0 |] in
  let upstream = Tinygrad_ml.Tensor.from_array [| 2.0; 3.0; 5.0; 7.0 |] in
  let y = Tinygrad_ml.Tensor.where_ cond t f in
  let grads = Tinygrad_ml.Tensor.backward ~grad:upstream ~wrt:[ cond; t; f ] y in
  let _, dcond = List.nth grads 0 in
  let _, dt = List.nth grads 1 in
  let _, df = List.nth grads 2 in
  let expected_dcond = [| 0.0; 0.0; 0.0; 0.0 |] in
  let expected_dt = [| 2.0; 0.0; 5.0; 0.0 |] in
  let expected_df = [| 0.0; 3.0; 0.0; 7.0 |] in
  let cpu_dcond = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c dcond in
  let cpu_dt = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c dt in
  let cpu_df = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c df in
  check_array ~msg:"cpu d/dcond where(cond,t,f)" expected_dcond cpu_dcond;
  check_array ~msg:"cpu d/dt where(cond,t,f)" expected_dt cpu_dt;
  check_array ~msg:"cpu d/df where(cond,t,f)" expected_df cpu_df;
  let metal_dcond = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Metal dcond in
  let metal_dt = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Metal dt in
  let metal_df = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Metal df in
  check_array ~msg:"metal d/dcond where(cond,t,f)" expected_dcond metal_dcond;
  check_array ~msg:"metal d/dt where(cond,t,f)" expected_dt metal_dt;
  check_array ~msg:"metal d/df where(cond,t,f)" expected_df metal_df

let run_or_skip () =
  match Tinygrad_ml.Metal_backend.available () with
  | Error msg ->
      Printf.printf "test_metal: skipped (%s)\n%!" msg;
      ()
  | Ok () ->
      test_add_mul_metal ();
      test_unary_chain_metal ();
      test_matches_cpu ();
      test_chain_realize_metal ();
      test_reductions_metal ();
      test_fused_reductions_metal_vs_cpu ();
      test_axis_reductions_metal_vs_cpu ();
      test_movement_ops_metal_vs_cpu ();
      test_where_cast_metal_vs_cpu ();
      test_where_backward_metal ();
      Printf.printf "test_metal: ok\n%!"

let () = run_or_skip ()

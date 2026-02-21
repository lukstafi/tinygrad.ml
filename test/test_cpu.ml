let check_close ~msg a b =
  let eps = max 1e-6 (1e-6 *. Float.abs a) in
  if Float.abs (a -. b) > eps then
    failwith (Printf.sprintf "%s: expected %.8f, got %.8f" msg a b)

let check_array ~msg expected got =
  if Array.length expected <> Array.length got then
    failwith
      (Printf.sprintf "%s: length mismatch expected=%d got=%d" msg (Array.length expected)
         (Array.length got));
  Array.iteri (fun i e -> check_close ~msg:(Printf.sprintf "%s[%d]" msg i) e got.(i)) expected

let test_add_mul_cpu () =
  let a = Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0; 4.0 |] in
  let b = Tinygrad_ml.Tensor.from_array [| 5.0; 6.0; 7.0; 8.0 |] in
  let expr = Tinygrad_ml.Tensor.mul (Tinygrad_ml.Tensor.add a b) b in
  let out = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c expr in
  check_array ~msg:"(a+b)*b" [| 30.0; 48.0; 70.0; 96.0 |] out

let test_sub_neg_sqrt_reciprocal_cpu () =
  let a = Tinygrad_ml.Tensor.from_array [| 4.0; 9.0; 16.0; 25.0 |] in
  let b = Tinygrad_ml.Tensor.from_array [| 1.0; 4.0; 9.0; 16.0 |] in
  let subbed = Tinygrad_ml.Tensor.sub a b in
  let neg_then_neg = Tinygrad_ml.Tensor.neg (Tinygrad_ml.Tensor.neg subbed) in
  let rooted = Tinygrad_ml.Tensor.sqrt neg_then_neg in
  let inv = Tinygrad_ml.Tensor.reciprocal rooted in
  let out = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c inv in
  check_array ~msg:"reciprocal(sqrt(a-b))"
    [| 1.0 /. sqrt 3.0; 1.0 /. sqrt 5.0; 1.0 /. sqrt 7.0; 1.0 /. 3.0 |]
    out

let test_exp2_log2_sin_cpu () =
  let x = Tinygrad_ml.Tensor.from_array [| 0.0; 1.0; 2.0 |] in
  let exp2_out = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c (Tinygrad_ml.Tensor.exp2 x) in
  check_array ~msg:"exp2" [| 1.0; 2.0; 4.0 |] exp2_out;
  let y = Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 4.0 |] in
  let log2_out = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c (Tinygrad_ml.Tensor.log2 y) in
  check_array ~msg:"log2" [| 0.0; 1.0; 2.0 |] log2_out;
  let pi = Float.pi in
  let z = Tinygrad_ml.Tensor.from_array [| 0.0; pi /. 2.0; pi |] in
  let sin_out = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c (Tinygrad_ml.Tensor.sin z) in
  check_array ~msg:"sin" [| 0.0; 1.0; 0.0 |] sin_out

let test_sum_cpu () =
  let x = Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0 |] in
  let s = Tinygrad_ml.Tensor.sum ~device:Tinygrad_ml.Runtime.Cpu_c x in
  check_close ~msg:"sum" 6.0 s

let test_max_mean_cpu () =
  let x = Tinygrad_ml.Tensor.from_array [| 2.0; 7.0; 5.0; 4.0 |] in
  let m = Tinygrad_ml.Tensor.max ~device:Tinygrad_ml.Runtime.Cpu_c x in
  check_close ~msg:"max" 7.0 m;
  let mean = Tinygrad_ml.Tensor.mean ~device:Tinygrad_ml.Runtime.Cpu_c x in
  check_close ~msg:"mean" 4.5 mean

let test_fused_reductions_cpu () =
  let a = Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0; 4.0 |] in
  let b = Tinygrad_ml.Tensor.from_array [| 10.0; 20.0; 30.0; 40.0 |] in
  let fused = Tinygrad_ml.Tensor.sqrt (Tinygrad_ml.Tensor.mul (Tinygrad_ml.Tensor.add a b) a) in
  let s = Tinygrad_ml.Tensor.sum ~device:Tinygrad_ml.Runtime.Cpu_c fused in
  let m = Tinygrad_ml.Tensor.max ~device:Tinygrad_ml.Runtime.Cpu_c fused in
  let expected = [| sqrt 11.0; sqrt 44.0; sqrt 99.0; sqrt 176.0 |] in
  let expected_sum = Array.fold_left ( +. ) 0.0 expected in
  let expected_max = Array.fold_left max Float.neg_infinity expected in
  check_close ~msg:"fused sum" expected_sum s;
  check_close ~msg:"fused max" expected_max m

let test_kernel_cache_for_fused_expr () =
  Tinygrad_ml.Cpu_c_backend.reset_kernel_cache_for_tests ();
  let before = Tinygrad_ml.Cpu_c_backend.compiled_kernel_count () in
  let a1 = Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0; 4.0 |] in
  let b1 = Tinygrad_ml.Tensor.from_array [| 2.0; 3.0; 4.0; 5.0 |] in
  let expr1 = Tinygrad_ml.Tensor.mul (Tinygrad_ml.Tensor.add a1 b1) b1 in
  ignore (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c expr1);
  let after_first = Tinygrad_ml.Cpu_c_backend.compiled_kernel_count () in
  if after_first <> before + 1 then
    failwith
      (Printf.sprintf "expected one compiled fused kernel, before=%d after_first=%d" before after_first);
  let a2 = Tinygrad_ml.Tensor.from_array [| 5.0; 6.0; 7.0; 8.0 |] in
  let b2 = Tinygrad_ml.Tensor.from_array [| 1.0; 1.5; 2.0; 2.5 |] in
  let expr2 = Tinygrad_ml.Tensor.mul (Tinygrad_ml.Tensor.add a2 b2) b2 in
  ignore (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c expr2);
  let after_second = Tinygrad_ml.Cpu_c_backend.compiled_kernel_count () in
  if after_second <> after_first then
    failwith
      (Printf.sprintf "expected compiled kernel cache hit, after_first=%d after_second=%d"
         after_first after_second)

let test_reshape_and_axis_reductions_cpu () =
  let a =
    Tinygrad_ml.Tensor.reshape
      (Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0; 7.0; 8.0 |])
      [| 2; 4 |]
  in
  let b =
    Tinygrad_ml.Tensor.reshape
      (Tinygrad_ml.Tensor.from_array [| 8.0; 7.0; 6.0; 5.0; 4.0; 3.0; 2.0; 1.0 |])
      [| 2; 4 |]
  in
  let c = Tinygrad_ml.Tensor.add a b in
  let c_arr = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c c in
  check_array ~msg:"reshape add" [| 9.0; 9.0; 9.0; 9.0; 9.0; 9.0; 9.0; 9.0 |] c_arr;
  let expr_reshaped = Tinygrad_ml.Tensor.reshape c [| 4; 2 |] in
  check_array ~msg:"reshape unrealized expression" [| 9.0; 9.0; 9.0; 9.0; 9.0; 9.0; 9.0; 9.0 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c expr_reshaped);
  let s = Tinygrad_ml.Tensor.sum_axis ~device:Tinygrad_ml.Runtime.Cpu_c ~axes:[ 1 ] a in
  check_array ~msg:"sum_axis axis=1" [| 10.0; 26.0 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c s);
  let m = Tinygrad_ml.Tensor.max_axis ~device:Tinygrad_ml.Runtime.Cpu_c ~axes:[ 0 ] a in
  check_array ~msg:"max_axis axis=0" [| 5.0; 6.0; 7.0; 8.0 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c m);
  let mean = Tinygrad_ml.Tensor.mean_axis ~device:Tinygrad_ml.Runtime.Cpu_c ~axes:[ 1 ] a in
  check_array ~msg:"mean_axis axis=1" [| 2.5; 6.5 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c mean)

let test_expand_cpu () =
  let a =
    Tinygrad_ml.Tensor.reshape
      (Tinygrad_ml.Tensor.from_array [| 2.0; 3.0; 4.0 |])
      [| 3; 1 |]
  in
  let a_exp = Tinygrad_ml.Tensor.expand a [| 3; 2 |] in
  check_array ~msg:"expand forward"
    [| 2.0; 2.0; 3.0; 3.0; 4.0; 4.0 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c a_exp);
  let b =
    Tinygrad_ml.Tensor.reshape
      (Tinygrad_ml.Tensor.from_array [| 1.0; 10.0; 1.0; 10.0; 1.0; 10.0 |])
      [| 3; 2 |]
  in
  let prod = Tinygrad_ml.Tensor.mul a_exp b in
  check_array ~msg:"expand in expression"
    [| 2.0; 20.0; 3.0; 30.0; 4.0; 40.0 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c prod)

let test_permute_cpu () =
  let a =
    Tinygrad_ml.Tensor.reshape
      (Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |])
      [| 2; 3 |]
  in
  let ap = Tinygrad_ml.Tensor.permute a [| 1; 0 |] in
  check_array ~msg:"permute forward"
    [| 1.0; 4.0; 2.0; 5.0; 3.0; 6.0 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c ap)

let test_noncontiguous_axis_reductions_cpu () =
  let a =
    Tinygrad_ml.Tensor.reshape
      (Tinygrad_ml.Tensor.from_array (Array.init 24 (fun i -> float_of_int (i + 1))))
      [| 2; 3; 4 |]
  in
  let s = Tinygrad_ml.Tensor.sum_axis ~device:Tinygrad_ml.Runtime.Cpu_c ~axes:[ 0; 2 ] a in
  check_array ~msg:"sum_axis axes=[0;2]" [| 68.0; 100.0; 132.0 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c s);
  let m = Tinygrad_ml.Tensor.max_axis ~device:Tinygrad_ml.Runtime.Cpu_c ~axes:[ 0; 2 ] a in
  check_array ~msg:"max_axis axes=[0;2]" [| 16.0; 20.0; 24.0 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c m);
  let mean = Tinygrad_ml.Tensor.mean_axis ~device:Tinygrad_ml.Runtime.Cpu_c ~axes:[ 0; 2 ] a in
  check_array ~msg:"mean_axis axes=[0;2]" [| 8.5; 12.5; 16.5 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c mean)

let test_reshape_preserves_realize_cache_cpu () =
  Tinygrad_ml.Cpu_c_backend.reset_kernel_cache_for_tests ();
  let a = Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0; 4.0 |] in
  let b = Tinygrad_ml.Tensor.from_array [| 10.0; 20.0; 30.0; 40.0 |] in
  let expr = Tinygrad_ml.Tensor.add a b in
  ignore (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c expr);
  let after_first = Tinygrad_ml.Cpu_c_backend.compiled_kernel_count () in
  let reshaped = Tinygrad_ml.Tensor.reshape expr [| 2; 2 |] in
  ignore (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c reshaped);
  let after_second = Tinygrad_ml.Cpu_c_backend.compiled_kernel_count () in
  if after_second <> after_first then
    failwith
      (Printf.sprintf "reshape should reuse realized cache, after_first=%d after_second=%d"
         after_first after_second)

let test_backward_basic_cpu () =
  let x = Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0 |] in
  let y = Tinygrad_ml.Tensor.mul x x in
  let grads = Tinygrad_ml.Tensor.backward ~wrt:[ x ] y in
  let _, dx = List.hd grads in
  check_array ~msg:"d/dx sum(x*x)" [| 2.0; 4.0; 6.0 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c dx);
  let a = Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0; 4.0 |] in
  let b = Tinygrad_ml.Tensor.from_array [| 10.0; 20.0; 30.0; 40.0 |] in
  let z = Tinygrad_ml.Tensor.mul a b in
  let grads2 = Tinygrad_ml.Tensor.backward ~wrt:[ a; b ] z in
  let _, da = List.nth grads2 0 in
  let _, db = List.nth grads2 1 in
  check_array ~msg:"d/da sum(a*b)" [| 10.0; 20.0; 30.0; 40.0 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c da);
  check_array ~msg:"d/db sum(a*b)" [| 1.0; 2.0; 3.0; 4.0 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c db)

let test_gradient_descent_cpu () =
  let target = [| 3.0; 5.0 |] in
  let x = ref [| 0.0; 0.0 |] in
  let lr = 0.1 in
  for _step = 0 to 29 do
    let xt = Tinygrad_ml.Tensor.from_array !x in
    let tt = Tinygrad_ml.Tensor.from_array target in
    let diff = Tinygrad_ml.Tensor.sub xt tt in
    let sq = Tinygrad_ml.Tensor.mul diff diff in
    let grads = Tinygrad_ml.Tensor.backward ~wrt:[ xt ] sq in
    let _, dx = List.hd grads in
    let dx_arr = Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c dx in
    for i = 0 to Array.length !x - 1 do
      (!x).(i) <- (!x).(i) -. (lr *. dx_arr.(i))
    done
  done;
  if Float.abs ((!x).(0) -. 3.0) > 0.05 then
    failwith (Printf.sprintf "gd x[0]: expected ~3.0, got %.8f" (!x).(0));
  if Float.abs ((!x).(1) -. 5.0) > 0.05 then
    failwith (Printf.sprintf "gd x[1]: expected ~5.0, got %.8f" (!x).(1))

let test_backward_reductions_cpu () =
  let x =
    Tinygrad_ml.Tensor.reshape
      (Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |])
      [| 2; 3 |]
  in
  let y = Tinygrad_ml.Tensor.sum_axis ~axes:[ 1 ] (Tinygrad_ml.Tensor.mul x x) in
  let grads = Tinygrad_ml.Tensor.backward ~wrt:[ x ] y in
  let _, dx = List.hd grads in
  check_array ~msg:"d/dx sum_axis(x*x,axis=1)" [| 2.0; 4.0; 6.0; 8.0; 10.0; 12.0 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c dx);
  let y_max = Tinygrad_ml.Tensor.max_axis ~axes:[ 1 ] x in
  let grads_max = Tinygrad_ml.Tensor.backward ~wrt:[ x ] y_max in
  let _, dx_max = List.hd grads_max in
  check_array ~msg:"d/dx max_axis(x,axis=1) unique maxima mask" [| 0.0; 0.0; 1.0; 0.0; 0.0; 1.0 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c dx_max);
  let x_tie = Tinygrad_ml.Tensor.from_array [| 3.0; 7.0; 7.0; 2.0 |] in
  let y_tie = Tinygrad_ml.Tensor.max_axis ~axes:[ 0 ] x_tie in
  let grads_tie = Tinygrad_ml.Tensor.backward ~wrt:[ x_tie ] y_tie in
  let _, dx_tie = List.hd grads_tie in
  check_array ~msg:"d/dx max_axis tie split" [| 0.0; 0.5; 0.5; 0.0 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c dx_tie);
  let y_mean = Tinygrad_ml.Tensor.mean_axis ~axes:[ 1 ] x in
  let grads_mean = Tinygrad_ml.Tensor.backward ~wrt:[ x ] y_mean in
  let _, dx_mean = List.hd grads_mean in
  check_array ~msg:"d/dx mean_axis(x,axis=1)" [| 1.0 /. 3.0; 1.0 /. 3.0; 1.0 /. 3.0; 1.0 /. 3.0; 1.0 /. 3.0; 1.0 /. 3.0 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c dx_mean);
  let reduced = Tinygrad_ml.Tensor.sum_axis ~axes:[ 1 ] x in
  let one = Tinygrad_ml.Tensor.reshape (Tinygrad_ml.Tensor.ones 2) [| 2; 1 |] in
  let expr = Tinygrad_ml.Tensor.add reduced one in
  check_array ~msg:"reduce_axis intermediate forward" [| 7.0; 16.0 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c expr);
  let grads_expr = Tinygrad_ml.Tensor.backward ~wrt:[ x ] expr in
  let _, dx_expr = List.hd grads_expr in
  check_array ~msg:"reduce_axis intermediate backward" [| 1.0; 1.0; 1.0; 1.0; 1.0; 1.0 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c dx_expr)

let test_backward_expand_cpu () =
  let x =
    Tinygrad_ml.Tensor.reshape
      (Tinygrad_ml.Tensor.from_array [| 3.0; 4.0 |])
      [| 2; 1 |]
  in
  let y = Tinygrad_ml.Tensor.expand x [| 2; 3 |] in
  let grads = Tinygrad_ml.Tensor.backward ~wrt:[ x ] y in
  let _, dx = List.hd grads in
  check_array ~msg:"d/dx sum(expand(x))"
    [| 3.0; 3.0 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c dx);
  let sq = Tinygrad_ml.Tensor.mul y y in
  let grads2 = Tinygrad_ml.Tensor.backward ~wrt:[ x ] sq in
  let _, dx2 = List.hd grads2 in
  check_array ~msg:"d/dx sum(expand(x)^2)"
    [| 18.0; 24.0 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c dx2)

let test_backward_permute_cpu () =
  let x =
    Tinygrad_ml.Tensor.reshape
      (Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |])
      [| 2; 3 |]
  in
  let xp = Tinygrad_ml.Tensor.permute x [| 1; 0 |] in
  let grads = Tinygrad_ml.Tensor.backward ~wrt:[ x ] xp in
  let _, dx = List.hd grads in
  check_array ~msg:"d/dx sum(permute(x))"
    [| 1.0; 1.0; 1.0; 1.0; 1.0; 1.0 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c dx)

let test_backward_unary_extra_cpu () =
  let x = Tinygrad_ml.Tensor.from_array [| 0.0; 1.0; 2.0 |] in
  let y = Tinygrad_ml.Tensor.exp2 x in
  let grads = Tinygrad_ml.Tensor.backward ~wrt:[ x ] y in
  let _, dx = List.hd grads in
  let ln2 = Float.log 2.0 in
  check_array ~msg:"d/dx sum(exp2(x))" [| 1.0 *. ln2; 2.0 *. ln2; 4.0 *. ln2 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c dx);
  let z = Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 4.0 |] in
  let w = Tinygrad_ml.Tensor.log2 z in
  let grads2 = Tinygrad_ml.Tensor.backward ~wrt:[ z ] w in
  let _, dz = List.hd grads2 in
  check_array ~msg:"d/dx sum(log2(x))"
    [| 1.0 /. (1.0 *. ln2); 1.0 /. (2.0 *. ln2); 1.0 /. (4.0 *. ln2) |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c dz);
  let pi = Float.pi in
  let s = Tinygrad_ml.Tensor.from_array [| 0.0; pi /. 4.0; pi /. 2.0 |] in
  let t = Tinygrad_ml.Tensor.sin s in
  let grads3 = Tinygrad_ml.Tensor.backward ~wrt:[ s ] t in
  let _, ds = List.hd grads3 in
  check_array ~msg:"d/dx sum(sin(x))" [| 1.0; Float.sqrt 2.0 /. 2.0; 0.0 |]
    (Tinygrad_ml.Tensor.to_array ~device:Tinygrad_ml.Runtime.Cpu_c ds)

let () =
  test_add_mul_cpu ();
  test_sub_neg_sqrt_reciprocal_cpu ();
  test_exp2_log2_sin_cpu ();
  test_sum_cpu ();
  test_max_mean_cpu ();
  test_fused_reductions_cpu ();
  test_kernel_cache_for_fused_expr ();
  test_reshape_and_axis_reductions_cpu ();
  test_expand_cpu ();
  test_permute_cpu ();
  test_noncontiguous_axis_reductions_cpu ();
  test_reshape_preserves_realize_cache_cpu ();
  test_backward_basic_cpu ();
  test_gradient_descent_cpu ();
  test_backward_reductions_cpu ();
  test_backward_expand_cpu ();
  test_backward_permute_cpu ();
  test_backward_unary_extra_cpu ();
  Printf.printf "test_cpu: ok\n"

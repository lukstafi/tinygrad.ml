let check_close ~msg a b =
  let eps = 1e-6 in
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

let () =
  test_add_mul_cpu ();
  test_sub_neg_sqrt_reciprocal_cpu ();
  test_sum_cpu ();
  test_max_mean_cpu ();
  test_kernel_cache_for_fused_expr ();
  Printf.printf "test_cpu: ok\n"

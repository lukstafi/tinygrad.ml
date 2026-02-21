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

let test_sum_cpu () =
  let x = Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0 |] in
  let s = Tinygrad_ml.Tensor.sum ~device:Tinygrad_ml.Runtime.Cpu_c x in
  check_close ~msg:"sum" 6.0 s

let () =
  test_add_mul_cpu ();
  test_sum_cpu ();
  Printf.printf "test_cpu: ok\n"

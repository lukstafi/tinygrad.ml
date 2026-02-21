let check_close ~msg a b =
  let eps = 1e-5 in
  if Float.abs (a -. b) > eps then
    failwith (Printf.sprintf "%s: expected %.8f, got %.8f" msg a b)

let check_array ~msg expected got =
  if Array.length expected <> Array.length got then
    failwith
      (Printf.sprintf "%s: length mismatch expected=%d got=%d" msg (Array.length expected)
         (Array.length got));
  Array.iteri (fun i e -> check_close ~msg:(Printf.sprintf "%s[%d]" msg i) e got.(i)) expected

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

let run_or_skip () =
  match Tinygrad_ml.Metal_backend.available () with
  | Error msg ->
      Printf.printf "test_metal: skipped (%s)\n%!" msg;
      ()
  | Ok () ->
      test_add_mul_metal ();
      test_unary_chain_metal ();
      test_matches_cpu ();
      Printf.printf "test_metal: ok\n%!"

let () = run_or_skip ()

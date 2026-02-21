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
      Printf.printf "test_cuda: ok\n%!"

let () = run_or_skip ()

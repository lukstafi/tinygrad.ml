let check_float name got expect tol =
  if Float.abs (got -. expect) < tol then Printf.printf "  PASS %s (%.6f)\n%!" name got
  else (Printf.printf "  FAIL %s: got %.6f expected %.6f\n%!" name got expect; exit 1)

let () =
  (* Matmul backward bug: d/dw of sum(matmul(x,w) * mask)
     x = [[1,2],[3,4]], w = [[0.1,0.1],[0.1,0.1]]
     matmul = [[0.3,0.3],[0.7,0.7]]
     mask = [[1,0],[0,0]], loss = 0.3
     Expected dw = [[1,0],[2,0]] (x[0]^T @ mask_row0)
     Actual: [4,0,0,0] *)
  Printf.printf "\n=== Matmul backward debug ===\n%!";

  (* First: decompose matmul manually and test each step *)
  Schedule.reset ();
  let x = Tensor.from_float_list [2; 2] [1.;2.; 3.;4.] in
  let w = Tensor.from_float_list [2; 2] [0.1;0.1; 0.1;0.1] in

  (* Step 1: reshape + expand for x *)
  let x3 = Tensor.reshape x [2; 2; 1] in
  let x_exp = Tensor.expand x3 [2; 2; 2] in
  (* Step 2: reshape + expand for w *)
  let w3 = Tensor.reshape w [1; 2; 2] in
  let w_exp = Tensor.expand w3 [2; 2; 2] in
  (* Step 3: elementwise mul *)
  let prod = Tensor.mul x_exp w_exp in
  (* Step 4: sum over K=axis 1 *)
  let summed = Tensor.sum ~axes:[1] prod in  (* [2;1;2] *)
  (* Step 5: reshape to [2;2] *)
  let result = Tensor.reshape summed [2; 2] in

  (* Verify forward *)
  let rv = Tensor.to_float_list result in
  Printf.printf "  matmul result: [%s]\n%!" (String.concat "; " (List.map (Printf.sprintf "%.4f") rv));
  check_float "matmul[0][0]" (List.nth rv 0) 0.3 1e-5;

  (* Apply mask and compute loss *)
  Schedule.reset ();
  let x2 = Tensor.from_float_list [2; 2] [1.;2.; 3.;4.] in
  let w2 = Tensor.from_float_list [2; 2] [0.1;0.1; 0.1;0.1] in
  let x2_3 = Tensor.reshape x2 [2; 2; 1] in
  let x2_exp = Tensor.expand x2_3 [2; 2; 2] in
  let w2_3 = Tensor.reshape w2 [1; 2; 2] in
  let w2_exp = Tensor.expand w2_3 [2; 2; 2] in
  let prod2 = Tensor.mul x2_exp w2_exp in
  let summed2 = Tensor.sum ~axes:[1] prod2 in
  let result2 = Tensor.reshape summed2 [2; 2] in
  let mask = Tensor.from_float_list [2; 2] [1.;0.;0.;0.] in
  let masked = Tensor.mul result2 mask in
  let loss = Tensor.sum masked in

  let loss_v = Tensor.to_float_list loss in
  Printf.printf "  loss = %.4f\n%!" (List.hd loss_v);

  (* Backward w.r.t. w *)
  let grads = Tensor.backward loss [w2] in
  let (_, dw) = List.hd grads in
  let dw_v = Tensor.to_float_list dw in
  Printf.printf "  dw: [%s]\n%!" (String.concat "; " (List.map (Printf.sprintf "%.4f") dw_v));

  (* Also test backward w.r.t. x *)
  Schedule.reset ();
  let x3_ = Tensor.from_float_list [2; 2] [1.;2.; 3.;4.] in
  let w3_ = Tensor.from_float_list [2; 2] [0.1;0.1; 0.1;0.1] in
  let logits3 = Tensor.matmul x3_ w3_ in
  let mask3 = Tensor.from_float_list [2; 2] [1.;0.;0.;0.] in
  let loss3 = Tensor.sum (Tensor.mul logits3 mask3) in
  let grads3 = Tensor.backward loss3 [x3_] in
  let (_, dx3) = List.hd grads3 in
  let dx3_v = Tensor.to_float_list dx3 in
  Printf.printf "  dx: [%s]\n%!" (String.concat "; " (List.map (Printf.sprintf "%.4f") dx3_v));
  (* dx should be: d/dx[i][j] = sum_k(w[j][k] * mask[i][k])
     dx[0][0] = w[0][0]*1 + w[0][1]*0 = 0.1
     dx[0][1] = w[1][0]*1 + w[1][1]*0 = 0.1
     dx[1][0] = 0, dx[1][1] = 0 *)
  check_float "dx[0][0]" (List.nth dx3_v 0) 0.1 1e-5;
  check_float "dx[0][1]" (List.nth dx3_v 1) 0.1 1e-5;
  check_float "dx[1][0]" (List.nth dx3_v 2) 0.0 1e-5;
  check_float "dx[1][1]" (List.nth dx3_v 3) 0.0 1e-5;

  Printf.printf "\n  Expected dw = [1, 0, 2, 0]\n%!";
  check_float "dw[0][0]" (List.nth dw_v 0) 1.0 1e-4;
  check_float "dw[0][1]" (List.nth dw_v 1) 0.0 1e-4;
  check_float "dw[1][0]" (List.nth dw_v 2) 2.0 1e-4;
  check_float "dw[1][1]" (List.nth dw_v 3) 0.0 1e-4;

  Printf.printf "\nAll tests passed!\n%!"

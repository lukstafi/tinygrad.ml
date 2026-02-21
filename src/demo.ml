let () =
  let dev = Tinygrad_ml.Runtime.default_device () in
  let a = Tinygrad_ml.Tensor.from_array [| 1.0; 2.0; 3.0; 4.0 |] in
  let b = Tinygrad_ml.Tensor.from_array [| 5.0; 6.0; 7.0; 8.0 |] in
  let expr =
    Tinygrad_ml.Tensor.reciprocal
      (Tinygrad_ml.Tensor.sqrt
         (Tinygrad_ml.Tensor.mul (Tinygrad_ml.Tensor.add a b) b))
  in
  let out = Tinygrad_ml.Tensor.to_array ~device:dev expr in
  let shown = Array.to_list out |> List.map string_of_float |> String.concat ", " in
  Printf.printf "device=%s\noutput=[%s]\n"
    (Tinygrad_ml.Runtime.device_to_string dev)
    shown

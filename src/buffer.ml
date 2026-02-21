open Bigarray

type t = {
  shape : int array;
  data : (float, float32_elt, c_layout) Array1.t;
}

let copy_shape shape = Array.copy shape

let numel shape =
  if Array.length shape = 0 then 1
  else Array.fold_left (fun acc d -> acc * d) 1 shape

let create shape =
  let n = numel shape in
  { shape = copy_shape shape; data = Array1.create float32 c_layout n }

let of_array arr =
  let out = { shape = [| Array.length arr |]; data = Array1.create float32 c_layout (Array.length arr) } in
  Array.iteri (fun i v -> out.data.{i} <- v) arr;
  out

let to_array t =
  Array.init (Array1.dim t.data) (fun i -> t.data.{i})

let zeros shape =
  let out = create shape in
  Array1.fill out.data 0.0;
  out

let ones shape =
  let out = create shape in
  Array1.fill out.data 1.0;
  out

let same_shape a b =
  Array.length a.shape = Array.length b.shape
  && Array.for_all2 ( = ) a.shape b.shape

let pp_shape shape =
  let dims = Array.to_list shape |> List.map string_of_int |> String.concat "x" in
  "[" ^ dims ^ "]"

type node =
  | Data of Buffer.t
  | Const of float
  | Binop of Uop.binop * t * t
  | Unop of Uop.unop * t

and t = {
  shape : int array;
  node : node;
  mutable cache : (Runtime.device * Buffer.t) list;
}

let make_data b = { shape = Array.copy b.Buffer.shape; node = Data b; cache = [] }

let from_array arr = make_data (Buffer.of_array arr)

let full n value = { shape = [| n |]; node = Const value; cache = [] }
let zeros n = full n 0.0
let ones n = full n 1.0
let shape t = Array.copy t.shape
let numel t = Buffer.numel t.shape

let assert_same_shape a b =
  if Array.length a.shape <> Array.length b.shape || not (Array.for_all2 ( = ) a.shape b.shape) then
    invalid_arg
      (Printf.sprintf "shape mismatch in tensor op: %s vs %s"
         (Buffer.pp_shape a.shape)
         (Buffer.pp_shape b.shape))

let binop op a b =
  assert_same_shape a b;
  { shape = Array.copy a.shape; node = Binop (op, a, b); cache = [] }

let unop op a = { shape = Array.copy a.shape; node = Unop (op, a); cache = [] }

let add a b = binop Uop.Add a b
let sub a b = binop Uop.Sub a b
let mul a b = binop Uop.Mul a b
let neg a = unop Uop.Neg a
let sqrt a = unop Uop.Sqrt a
let reciprocal a = unop Uop.Reciprocal a

let find_cache dev entries =
  List.find_map (fun (d, b) -> if d = dev then Some b else None) entries

let update_cache dev buf entries =
  (dev, buf) :: List.filter (fun (d, _) -> d <> dev) entries

let rec lower_to_expr t inputs =
  match t.node with
  | Data b ->
      let idx = List.length inputs in
      (Uop.Input idx, inputs @ [ b ])
  | Const c -> (Uop.Const c, inputs)
  | Binop (op, lhs, rhs) ->
      let lhs_expr, inputs = lower_to_expr lhs inputs in
      let rhs_expr, inputs = lower_to_expr rhs inputs in
      (Uop.Binop (op, lhs_expr, rhs_expr), inputs)
  | Unop (op, x) ->
      let x_expr, inputs = lower_to_expr x inputs in
      (Uop.Unop (op, x_expr), inputs)

let realize_result ?device t =
  let dev = Option.value device ~default:(Runtime.default_device ()) in
  match find_cache dev t.cache with
  | Some b -> Ok b
  | None ->
      let computed =
        match t.node with
        | Data buf -> Ok buf
        | _ ->
            let expr, inputs = lower_to_expr t [] in
            Runtime.run_expr ~device:dev ~expr ~inputs ~shape:t.shape
      in
      (match computed with
      | Ok buf ->
          t.cache <- update_cache dev buf t.cache;
          Ok buf
      | Error _ as e -> e)

let realize ?device t =
  match realize_result ?device t with
  | Ok b -> b
  | Error msg -> failwith msg

let to_array ?device t = Buffer.to_array (realize ?device t)

let sum ?device t =
  Array.fold_left ( +. ) 0.0 (to_array ?device t)

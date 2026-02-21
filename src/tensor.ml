type node =
  | Data of Buffer.t
  | Binop of Uop.binop * t * t

and t = {
  shape : int array;
  node : node;
  mutable cached : (Runtime.device * Buffer.t) option;
}

let make_data b = { shape = Array.copy b.Buffer.shape; node = Data b; cached = None }

let from_array arr = make_data (Buffer.of_array arr)
let zeros n = make_data (Buffer.zeros [| n |])
let ones n = make_data (Buffer.ones [| n |])
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
  { shape = Array.copy a.shape; node = Binop (op, a, b); cached = None }

let add a b = binop Uop.Add a b
let mul a b = binop Uop.Mul a b

let rec realize_result ?device t =
  let dev = Option.value device ~default:(Runtime.default_device ()) in
  match t.cached with
  | Some (cached_dev, buf) when cached_dev = dev -> Ok buf
  | _ ->
      let computed =
        match t.node with
        | Data buf -> Ok buf
        | Binop (op, lhs, rhs) ->
            let* lbuf = realize_result ~device:dev lhs in
            let* rbuf = realize_result ~device:dev rhs in
            Runtime.run_binop ~device:dev ~op ~a:lbuf ~b:rbuf
      in
      (match computed with
      | Ok buf ->
          t.cached <- Some (dev, buf);
          Ok buf
      | Error _ as e -> e)

and ( let* ) r f = match r with Ok x -> f x | Error _ as e -> e

let realize ?device t =
  match realize_result ?device t with
  | Ok b -> b
  | Error msg -> failwith msg

let to_array ?device t = Buffer.to_array (realize ?device t)

let sum ?device t =
  Array.fold_left ( +. ) 0.0 (to_array ?device t)

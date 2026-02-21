type node =
  | Data of Buffer.t
  | Const of float
  | Binop of Uop.binop * t * t
  | Unop of Uop.unop * t
  | Reshape of t
  | Expand of t
  | Reduce_axis of {
      op : Uop.reduce_op;
      axes : int list;
      src : t;
      device_hint : Runtime.device option;
    }

and t = {
  shape : int array;
  node : node;
  mutable cache : (Runtime.device * Buffer.t) list;
}

let make_data b = { shape = Array.copy b.Buffer.shape; node = Data b; cache = [] }

let from_array arr = make_data (Buffer.of_array arr)

let from_flat_array_with_shape (shape : int array) (arr : float array) =
  let expected = Buffer.numel shape in
  if Array.length arr <> expected then
    invalid_arg
      (Printf.sprintf "from_flat_array_with_shape: data length %d does not match shape numel %d"
         (Array.length arr) expected);
  let b = Buffer.create shape in
  Array.iteri (fun i v -> b.data.{i} <- v) arr;
  make_data b

let full n value = { shape = [| n |]; node = Const value; cache = [] }
let full_with_shape shape value = { shape = Array.copy shape; node = Const value; cache = [] }
let zeros n = full n 0.0
let ones n = full n 1.0
let zeros_like t = full_with_shape t.shape 0.0
let ones_like t = full_with_shape t.shape 1.0
let shape t = Array.copy t.shape
let numel t = Buffer.numel t.shape

let assert_same_shape a b =
  if Array.length a.shape <> Array.length b.shape || not (Array.for_all2 ( = ) a.shape b.shape) then
    invalid_arg
      (Printf.sprintf "shape mismatch in tensor op: %s vs %s"
         (Buffer.pp_shape a.shape)
         (Buffer.pp_shape b.shape))

let same_shape_arrays (a : int array) (b : int array) =
  Array.length a = Array.length b && Array.for_all2 ( = ) a b

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
let exp2 a = unop Uop.Exp2 a
let log2 a = unop Uop.Log2 a
let sin a = unop Uop.Sin a

let reshape t new_shape =
  if Buffer.numel new_shape <> numel t then
    invalid_arg
      (Printf.sprintf "reshape: numel mismatch %s -> %s"
         (Buffer.pp_shape t.shape) (Buffer.pp_shape new_shape));
  let cache =
    List.map
      (fun (dev, (b : Buffer.t)) -> (dev, { b with shape = Array.copy new_shape }))
      t.cache
  in
  { shape = Array.copy new_shape; node = Reshape t; cache }

let expand t new_shape =
  if Array.length t.shape <> Array.length new_shape then
    invalid_arg
      (Printf.sprintf "expand: rank mismatch %s -> %s"
         (Buffer.pp_shape t.shape) (Buffer.pp_shape new_shape));
  Array.iteri
    (fun i d ->
      let s = t.shape.(i) in
      if s <> 1 && s <> d then
        invalid_arg
          (Printf.sprintf "expand: dim %d incompatible (%d -> %d), expected source dim 1 or %d"
             i s d d))
    new_shape;
  { shape = Array.copy new_shape; node = Expand t; cache = [] }

let find_cache dev entries =
  List.find_map (fun (d, b) -> if d = dev then Some b else None) entries

let update_cache dev buf entries =
  (dev, buf) :: List.filter (fun (d, _) -> d <> dev) entries

let normalize_axes (shape : int array) (axes : int list) =
  let ndim = Array.length shape in
  let normalized =
    List.map
      (fun ax ->
        let a = if ax < 0 then ndim + ax else ax in
        if a < 0 || a >= ndim then
          invalid_arg
            (Printf.sprintf "axis %d out of bounds for tensor with ndim=%d" ax ndim);
        a)
      axes
  in
  List.sort_uniq compare normalized

let compute_strides (shape : int array) =
  let n = Array.length shape in
  let strides = Array.make n 1 in
  for i = n - 2 downto 0 do
    strides.(i) <- strides.(i + 1) * shape.(i + 1)
  done;
  strides

let expand_host_data ~(src_arr : float array) ~(src_shape : int array) ~(out_shape : int array) =
  let ndim = Array.length src_shape in
  if Array.length out_shape <> ndim then
    invalid_arg
      (Printf.sprintf "expand_host_data: rank mismatch %s -> %s"
         (Buffer.pp_shape src_shape) (Buffer.pp_shape out_shape));
  Array.iteri
    (fun i out_d ->
      let src_d = src_shape.(i) in
      if src_d <> 1 && src_d <> out_d then
        invalid_arg
          (Printf.sprintf "expand_host_data: incompatible dim %d (%d -> %d)"
             i src_d out_d))
    out_shape;
  let src_strides = compute_strides src_shape in
  let out_strides = compute_strides out_shape in
  let out = Array.make (Buffer.numel out_shape) 0.0 in
  for out_idx = 0 to Array.length out - 1 do
    let rem = ref out_idx in
    let src_idx = ref 0 in
    for d = 0 to ndim - 1 do
      let coord = !rem / out_strides.(d) in
      rem := !rem mod out_strides.(d);
      let src_coord = if src_shape.(d) = 1 then 0 else coord in
      src_idx := !src_idx + (src_coord * src_strides.(d))
    done;
    out.(out_idx) <- src_arr.(!src_idx)
  done;
  out

let rec lower_to_expr_with_shape ~device t current_shape inputs =
  match t.node with
  | Data b ->
      let b_view =
        if Array.length b.shape = Array.length current_shape
           && Array.for_all2 ( = ) b.shape current_shape
        then b
        else { b with shape = Array.copy current_shape }
      in
      let idx = List.length inputs in
      (Uop.Input idx, inputs @ [ b_view ])
  | Const c -> (Uop.Const c, inputs)
  | Binop (op, lhs, rhs) ->
      let lhs_expr, inputs = lower_to_expr_with_shape ~device lhs current_shape inputs in
      let rhs_expr, inputs = lower_to_expr_with_shape ~device rhs current_shape inputs in
      (Uop.Binop (op, lhs_expr, rhs_expr), inputs)
  | Unop (op, x) ->
      let x_expr, inputs = lower_to_expr_with_shape ~device x current_shape inputs in
      (Uop.Unop (op, x_expr), inputs)
  | Reshape inner ->
      lower_to_expr_with_shape ~device inner current_shape inputs
  | Expand _ | Reduce_axis _ ->
      let b : Buffer.t = realize ~device t in
      let b_view =
        if Array.length b.shape = Array.length current_shape
           && Array.for_all2 ( = ) b.shape current_shape
        then b
        else { b with shape = Array.copy current_shape }
      in
      let idx = List.length inputs in
      (Uop.Input idx, inputs @ [ b_view ])

and lower_to_expr ~device t inputs =
  lower_to_expr_with_shape ~device t t.shape inputs

and realize_result ?device t =
  let dev = Option.value device ~default:(Runtime.default_device ()) in
  match find_cache dev t.cache with
  | Some b -> Ok b
  | None ->
      let computed =
        match t.node with
        | Data buf -> Ok buf
        | Expand src ->
            let src_arr = to_array ~device:dev src in
            let out = expand_host_data ~src_arr ~src_shape:src.shape ~out_shape:t.shape in
            let b = Buffer.create t.shape in
            Array.iteri (fun i v -> b.data.{i} <- v) out;
            Ok b
        | Reduce_axis { op; axes; src; device_hint } ->
            let reduce_dev = Option.value device ~default:(Option.value device_hint ~default:dev) in
            let out_shape, out = reduce_axis_host_data ~device:reduce_dev ~axes ~op src in
            let b = Buffer.create out_shape in
            Array.iteri (fun i v -> b.data.{i} <- v) out;
            Ok b
        | _ ->
            let expr, inputs = lower_to_expr ~device:dev t [] in
            Runtime.run_expr ~device:dev ~expr ~inputs ~shape:t.shape
      in
      (match computed with
      | Ok buf ->
          t.cache <- update_cache dev buf t.cache;
          Ok buf
      | Error _ as e -> e)

and realize ?device t =
  match realize_result ?device t with
  | Ok b -> b
  | Error msg -> failwith msg

and to_array ?device t = Buffer.to_array (realize ?device t)

and reduce_axis_host_data ?device ~(axes : int list) ~(op : Uop.reduce_op) t =
  let in_arr = to_array ?device t in
  let in_shape = t.shape in
  let ndim = Array.length in_shape in
  let axes = if axes = [] then List.init ndim Fun.id else normalize_axes in_shape axes in
  let is_reduced = Array.make ndim false in
  List.iter (fun a -> is_reduced.(a) <- true) axes;
  let out_shape = Array.mapi (fun i d -> if is_reduced.(i) then 1 else d) in_shape in
  let in_strides = compute_strides in_shape in
  let out_strides = compute_strides out_shape in
  let out_numel = Buffer.numel out_shape in
  let out =
    match op with
    | Uop.Sum -> Array.make out_numel 0.0
    | Uop.Max -> Array.make out_numel Float.neg_infinity
  in
  for idx = 0 to Array.length in_arr - 1 do
    let rem = ref idx in
    let out_idx = ref 0 in
    for d = 0 to ndim - 1 do
      let coord = !rem / in_strides.(d) in
      rem := !rem mod in_strides.(d);
      if not is_reduced.(d) then
        out_idx := !out_idx + (coord * out_strides.(d))
    done;
    begin
      match op with
      | Uop.Sum -> out.(!out_idx) <- out.(!out_idx) +. in_arr.(idx)
      | Uop.Max -> out.(!out_idx) <- max out.(!out_idx) in_arr.(idx)
    end
  done;
  (out_shape, out)

let reduce_axis_host ?device ~(axes : int list) ~(op : Uop.reduce_op) t =
  let out_shape, out = reduce_axis_host_data ?device ~axes ~op t in
  from_flat_array_with_shape out_shape out

let reduce_axis ?device ~(axes : int list) ~(op : Uop.reduce_op) t =
  let ndim = Array.length t.shape in
  let axes = if axes = [] then List.init ndim Fun.id else normalize_axes t.shape axes in
  let out_shape = Array.mapi (fun i d -> if List.mem i axes then 1 else d) t.shape in
  match device with
  | Some dev -> reduce_axis_host ~device:dev ~axes ~op t
  | None -> { shape = out_shape; node = Reduce_axis { op; axes; src = t; device_hint = None }; cache = [] }

let sum_axis ?device ~axes t = reduce_axis ?device ~axes ~op:Uop.Sum t
let max_axis ?device ~axes t = reduce_axis ?device ~axes ~op:Uop.Max t
let mean_axis ?device ~axes t =
  let s = sum_axis ?device ~axes t in
  let ndim = Array.length t.shape in
  let axes = if axes = [] then List.init ndim Fun.id else normalize_axes t.shape axes in
  let factor = List.fold_left (fun acc a -> acc * t.shape.(a)) 1 axes in
  if factor = 0 then failwith "mean_axis: invalid zero reduction factor";
  let inv = full_with_shape s.shape (1.0 /. float_of_int factor) in
  mul s inv

let reduce_scalar_result ?device ~(op : Uop.reduce_op) t =
  let dev = Option.value device ~default:(Runtime.default_device ()) in
  let expr, inputs = lower_to_expr ~device:dev t [] in
  match Runtime.run_reduce ~device:dev ~op ~expr ~inputs ~shape:t.shape with
  | Ok v -> v
  | Error msg -> failwith msg

let sum ?device t = reduce_scalar_result ?device ~op:Uop.Sum t

let max ?device t = reduce_scalar_result ?device ~op:Uop.Max t

let mean ?device t =
  let n = numel t in
  if n = 0 then failwith "mean of empty tensor is undefined"
  else
    sum ?device t /. float_of_int n

let children t =
  match t.node with
  | Data _ | Const _ -> []
  | Binop (_, a, b) -> [ a; b ]
  | Unop (_, x) -> [ x ]
  | Reshape x -> [ x ]
  | Expand x -> [ x ]
  | Reduce_axis { src; _ } -> [ src ]

let rec contains_phys x = function
  | [] -> false
  | y :: ys -> (x == y) || contains_phys x ys

let toposort root =
  let visited = ref [] in
  let order = ref [] in
  let rec dfs t =
    if not (contains_phys t !visited) then begin
      visited := t :: !visited;
      List.iter dfs (children t);
      order := t :: !order
    end
  in
  dfs root;
  List.rev !order

let find_grad grads t =
  List.find_map (fun (u, g) -> if u == t then Some g else None) !grads

let add_grad grads t g =
  let rec loop acc = function
    | [] -> List.rev ((t, g) :: acc)
    | ((u, existing) as pair) :: tl ->
        if u == t then List.rev_append acc ((u, add existing g) :: tl)
        else loop (pair :: acc) tl
  in
  grads := loop [] !grads

let reduce_out_shape src_shape axes =
  Array.mapi (fun i d -> if List.mem i axes then 1 else d) src_shape

let reduce_output_index ~src_shape ~reduced_shape ~axes idx =
  let ndim = Array.length src_shape in
  let src_strides = compute_strides src_shape in
  let reduced_strides = compute_strides reduced_shape in
  let is_reduced = Array.make ndim false in
  List.iter (fun a -> is_reduced.(a) <- true) axes;
  let rem = ref idx in
  let reduced_idx = ref 0 in
  for d = 0 to ndim - 1 do
    let coord = !rem / src_strides.(d) in
    rem := !rem mod src_strides.(d);
    if not is_reduced.(d) then
      reduced_idx := !reduced_idx + (coord * reduced_strides.(d))
  done;
  !reduced_idx

let sum_reduce_grad ~upstream ~src_shape ~axes =
  let reduced_shape = reduce_out_shape src_shape axes in
  let up_arr = to_array upstream in
  if Array.length upstream.shape <> Array.length reduced_shape
     || not (Array.for_all2 ( = ) upstream.shape reduced_shape)
  then
    invalid_arg
      (Printf.sprintf "sum_reduce_grad: upstream shape %s does not match reduced shape %s"
         (Buffer.pp_shape upstream.shape) (Buffer.pp_shape reduced_shape));
  let out = Array.make (Buffer.numel src_shape) 0.0 in
  for idx = 0 to Array.length out - 1 do
    let ridx = reduce_output_index ~src_shape ~reduced_shape ~axes idx in
    out.(idx) <- up_arr.(ridx)
  done;
  from_flat_array_with_shape src_shape out

let max_reduce_grad ~upstream ~src ~reduced ~axes =
  let src_shape = src.shape in
  let reduced_shape = reduce_out_shape src_shape axes in
  let src_arr = to_array src in
  let red_arr = to_array reduced in
  let up_arr = to_array upstream in
  if Array.length up_arr <> Array.length red_arr then
    invalid_arg
      (Printf.sprintf "max_reduce_grad: upstream length %d does not match reduced length %d"
         (Array.length up_arr) (Array.length red_arr));
  let tie_counts = Array.make (Array.length red_arr) 0 in
  for idx = 0 to Array.length src_arr - 1 do
    let ridx = reduce_output_index ~src_shape ~reduced_shape ~axes idx in
    if src_arr.(idx) = red_arr.(ridx) then tie_counts.(ridx) <- tie_counts.(ridx) + 1
  done;
  let out = Array.make (Array.length src_arr) 0.0 in
  for idx = 0 to Array.length src_arr - 1 do
    let ridx = reduce_output_index ~src_shape ~reduced_shape ~axes idx in
    let ties = tie_counts.(ridx) in
    if src_arr.(idx) = red_arr.(ridx) && ties > 0 then
      out.(idx) <- up_arr.(ridx) /. float_of_int ties
  done;
  from_flat_array_with_shape src_shape out

let local_grads t upstream =
  match t.node with
  | Data _ | Const _ -> []
  | Reshape x -> [ (x, reshape upstream x.shape) ]
  | Expand x ->
      let expanded_axes = ref [] in
      for i = 0 to Array.length x.shape - 1 do
        if x.shape.(i) = 1 && t.shape.(i) > 1 then expanded_axes := i :: !expanded_axes
      done;
      let axes = List.rev !expanded_axes in
      let g = if axes = [] then upstream else sum_axis ~axes upstream in
      [ (x, if same_shape_arrays g.shape x.shape then g else reshape g x.shape) ]
  | Reduce_axis { op; axes; src; _ } ->
      let grad_src =
        match op with
        | Uop.Sum -> sum_reduce_grad ~upstream ~src_shape:src.shape ~axes
        | Uop.Max -> max_reduce_grad ~upstream ~src ~reduced:t ~axes
      in
      [ (src, grad_src) ]
  | Unop (Uop.Neg, x) -> [ (x, neg upstream) ]
  | Unop (Uop.Sqrt, x) ->
      let two = full_with_shape t.shape 2.0 in
      let denom = mul two t in
      [ (x, mul upstream (reciprocal denom)) ]
  | Unop (Uop.Reciprocal, x) ->
      let scale = neg (mul t t) in
      [ (x, mul scale upstream) ]
  | Unop (Uop.Exp2, x) ->
      let ln2 = full_with_shape t.shape (Float.log 2.0) in
      [ (x, mul upstream (mul t ln2)) ]
  | Unop (Uop.Log2, x) ->
      let ln2 = full_with_shape x.shape (Float.log 2.0) in
      [ (x, mul upstream (reciprocal (mul x ln2))) ]
  | Unop (Uop.Sin, x) ->
      let half_pi = full_with_shape x.shape (Float.pi /. 2.0) in
      let cos_x = sin (sub half_pi x) in
      [ (x, mul upstream cos_x) ]
  | Binop (Uop.Add, a, b) -> [ (a, upstream); (b, upstream) ]
  | Binop (Uop.Sub, a, b) -> [ (a, upstream); (b, neg upstream) ]
  | Binop (Uop.Mul, a, b) -> [ (a, mul upstream b); (b, mul upstream a) ]

let backward ?grad ~wrt output =
  let root_grad =
    match grad with
    | Some g ->
        if Array.length g.shape <> Array.length output.shape
           || not (Array.for_all2 ( = ) g.shape output.shape)
        then
          invalid_arg
            (Printf.sprintf "backward: grad shape %s does not match output shape %s"
               (Buffer.pp_shape g.shape) (Buffer.pp_shape output.shape));
        g
    | None -> ones_like output
  in
  let topo = toposort output in
  let grads = ref [ (output, root_grad) ] in
  List.iter
    (fun t ->
      match find_grad grads t with
      | None -> ()
      | Some upstream ->
          List.iter (fun (src, gsrc) -> add_grad grads src gsrc) (local_grads t upstream))
    (List.rev topo);
  List.map
    (fun target ->
      let g = Option.value (find_grad grads target) ~default:(zeros_like target) in
      (target, g))
    wrt

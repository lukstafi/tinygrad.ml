(** Tensor: the user-facing API for tinygrad_ml.
    Ported from tinygrad/tensor.py (simplified).

    A Tensor wraps a UOp and tracks shape/dtype. All operations build the lazy
    computation graph; actual computation happens only when realize() is called.

    This is the minimal Tensor surface:
    - Creation: full, zeros, ones, from_float_list
    - Elementwise: +, -, *, /, neg, exp2, log2, sin, sqrt, reciprocal
    - Comparison: <, !=, ==, where
    - Movement: reshape, expand, permute, pad, shrink, flip
    - Reduction: sum, max, mean
    - Type: cast
    - Execution: realize, to_float_list *)

type t = {
  mutable uop: Uop.t;
  shape: int list;
  dtype: Dtype.t;
  device: string;
  requires_grad: bool;
}

(** Internal buffer counter for unique buffer IDs *)
let next_buf_id = ref 0
let fresh_buf_id () =
  let id = !next_buf_id in
  incr next_buf_id;
  id

(** Create a tensor from a pre-built UOp *)
let of_uop ?(device="CPU") ?(requires_grad=false) shape dtype uop =
  { uop; shape; dtype; device; requires_grad }

(** Tensor filled with a constant value *)
let full ?(device="CPU") ?(dtype=Dtype.float32) shape (value : float) =
  let numel = Helpers.prod shape in
  (* Create a CONST UOp, then RESHAPE to desired shape *)
  let c = Uop.const dtype value in
  let expanded = if numel > 1 then
    (* CONST → RESHAPE to (1,...) → EXPAND to shape *)
    let ones = List.map (fun _ -> 1) shape in
    let reshaped = Uop.reshape c ones in
    Uop.expand reshaped shape
  else c in
  of_uop ~device shape dtype expanded

let zeros ?(device="CPU") ?(dtype=Dtype.float32) shape = full ~device ~dtype shape 0.0
let ones ?(device="CPU") ?(dtype=Dtype.float32) shape = full ~device ~dtype shape 1.0

(** Create a tensor from a list of floats *)
let from_float_list ?(device="CPU") ?(dtype=Dtype.float32) shape (data : float list) =
  let numel = Helpers.prod shape in
  if List.length data <> numel then
    failwith (Printf.sprintf "data length %d doesn't match shape %s (numel=%d)"
      (List.length data) (String.concat "x" (List.map string_of_int shape)) numel);
  (* Create a buffer UOp and store the data in the schedule's buffer_data table *)
  let buf_id = fresh_buf_id () in
  let buf_uop = Uop.buffer buf_id (Dtype.ptr ~size:numel dtype) in
  (* Store data for later copyin during realize *)
  Schedule.store_buffer_data buf_uop.id (Array.of_list data);
  (* Build the graph: BUFFER → INDEX → LOAD → RESHAPE *)
  let idx = Uop.const Dtype.int32 0.0 in
  let indexed = Uop.index buf_uop idx in
  let loaded = Uop.load indexed in
  let reshaped = if List.length shape > 0 then Uop.reshape loaded shape else loaded in
  of_uop ~device shape dtype reshaped

(** Elementwise binary operations *)
let binop op (a : t) (b : t) =
  assert (a.shape = b.shape);
  let result_dtype = match op with
    | Ops.CMPLT | Ops.CMPNE | Ops.CMPEQ -> Dtype.bool
    | _ -> a.dtype
  in
  let uop = Uop.alu op a.dtype [a.uop; b.uop] in
  { a with uop; dtype = result_dtype }

(** Elementwise unary operations *)
let unop op (a : t) =
  let uop = Uop.alu op a.dtype [a.uop] in
  { a with uop }

let add a b = binop Ops.ADD a b
let sub a b = binop Ops.SUB a b
let mul a b = binop Ops.MUL a b
let div a b =
  let b_recip = unop Ops.RECIPROCAL b in
  mul a b_recip
let neg_ a = unop Ops.NEG a
let exp2_ a = unop Ops.EXP2 a
let log2_ a = unop Ops.LOG2 a
let sin_ a = unop Ops.SIN a
let sqrt_ a = unop Ops.SQRT a
let reciprocal a = unop Ops.RECIPROCAL a
let trunc_ a = unop Ops.TRUNC a

let lt a b = binop Ops.CMPLT a b
let ne a b = binop Ops.CMPNE a b
let eq a b = binop Ops.CMPEQ a b

let where_ cond t f =
  assert (cond.shape = t.shape && t.shape = f.shape);
  let uop = Uop.where_ cond.uop t.uop f.uop in
  { t with uop }

(** Scalar constant broadcast to tensor shape *)
let const_like (t : t) (v : float) =
  full ~device:t.device ~dtype:t.dtype t.shape v

(** Cast to a new dtype *)
let cast dtype (t : t) =
  let uop = Uop.cast dtype t.uop in
  { t with uop; dtype }

(** Movement operations *)
let reshape (t : t) new_shape =
  assert (Helpers.prod t.shape = Helpers.prod new_shape);
  let uop = Uop.reshape t.uop new_shape in
  { t with uop; shape = new_shape }

let expand (t : t) new_shape =
  (* each dim must be 1 or match *)
  assert (List.length t.shape = List.length new_shape);
  List.iter2 (fun s n -> assert (s = 1 || s = n)) t.shape new_shape;
  let uop = Uop.expand t.uop new_shape in
  { t with uop; shape = new_shape }

let permute (t : t) axes =
  assert (List.length axes = List.length t.shape);
  let new_shape = List.map (List.nth t.shape) axes in
  let uop = Uop.permute t.uop axes in
  { t with uop; shape = new_shape }

let pad (t : t) padding =
  assert (List.length padding = List.length t.shape);
  let new_shape = List.map2 (fun s (b, a) -> s + b + a) t.shape padding in
  let uop = Uop.pad t.uop padding in
  { t with uop; shape = new_shape }

let shrink (t : t) bounds =
  assert (List.length bounds = List.length t.shape);
  let new_shape = List.map (fun (lo, hi) -> hi - lo) bounds in
  let uop = Uop.shrink t.uop bounds in
  { t with uop; shape = new_shape }

let flip (t : t) axes =
  let uop = Uop.flip t.uop axes in
  { t with uop }

(** Reduction operations *)
let reduce op (t : t) axes =
  let new_shape = List.mapi (fun i s ->
    if List.mem i axes then 1 else s
  ) t.shape in
  let uop = Uop.reduce_axis t.uop op axes in
  { t with uop; shape = new_shape }

let sum ?(axes=[]) (t : t) =
  let axes = if axes = [] then List.init (List.length t.shape) Fun.id else axes in
  reduce Ops.ADD t axes

let max_ ?(axes=[]) (t : t) =
  let axes = if axes = [] then List.init (List.length t.shape) Fun.id else axes in
  reduce Ops.MAX t axes

let mean ?(axes=[]) (t : t) =
  let axes = if axes = [] then List.init (List.length t.shape) Fun.id else axes in
  let s = sum ~axes t in
  let n = List.fold_left (fun acc i -> acc * List.nth t.shape i) 1 axes in
  let n_inv = const_like s (1.0 /. Float.of_int n) in
  mul s n_inv

(** Contiguous *)
let contiguous (t : t) =
  let uop = Uop.contiguous t.uop in
  { t with uop }

(** Realize: trigger computation.
    This is where lazy evaluation ends and actual work begins.
    1. Schedule the computation graph
    2. Compile and execute kernels
    3. Replace tensor's UOp with a BUFFER reference to the realized data *)
let realize (t : t) =
  (* For reductions and mean (MUL over reduced result), we need to pass
     the input numel so the scheduler knows the reduction loop bound. *)
  let input_numel = match t.uop.op with
    | Ops.REDUCE_AXIS ->
      (* The source tensor had more elements; infer from BUFFER sizes in the graph *)
      let src_uops = Uop.toposort1 (List.hd t.uop.src) in
      List.fold_left (fun acc (u : Uop.t) ->
        match u.dtype with
        | Dtype.Ptr (_, _, sz) when sz > 0 -> max acc sz
        | _ -> acc
      ) (Helpers.prod t.shape) src_uops
    | _ ->
      (* Check if any child is a REDUCE_AXIS (e.g., mean = sum * 1/n) *)
      let all = Uop.toposort1 t.uop in
      let reduce_input = List.fold_left (fun acc (u : Uop.t) ->
        if u.op = Ops.REDUCE_AXIS then
          let src_uops = Uop.toposort1 (List.hd u.src) in
          let n = List.fold_left (fun a (su : Uop.t) ->
            match su.dtype with
            | Dtype.Ptr (_, _, sz) when sz > 0 -> max a sz
            | _ -> a
          ) 0 src_uops in
          max acc n
        else acc
      ) 0 all in
      if reduce_input > 0 then reduce_input else Helpers.prod t.shape
  in
  let schedule = Schedule.create_schedule ~device:t.device ~numel:(Helpers.prod t.shape) ~input_numel [t.uop] in
  Realize.run_schedule schedule;
  (* After realization, check if a result buffer was stored for this root UOp.
     If so, replace the tensor's UOp with a BUFFER node pointing to the result,
     so that subsequent operations referencing this tensor can find the data. *)
  (match Schedule.get_realized t.uop.id with
   | Some _buf ->
     let buf_id = fresh_buf_id () in
     let buf_uop = Uop.buffer buf_id (Dtype.ptr ~size:(Helpers.prod t.shape) t.dtype) in
     Schedule.store_realized buf_uop.id _buf;
     t.uop <- buf_uop
   | None -> ());
  t

(** Find the realized Device.buffer for this tensor.
    First checks if the root UOp itself was realized (computed result),
    then falls back to looking for BUFFER nodes (input data). *)
let find_realized_buffer (t : t) : Device.buffer option =
  (* Check if the root UOp was realized (result of a kernel computation) *)
  match Schedule.get_realized t.uop.id with
  | Some buf -> Some buf
  | None ->
    (* Fall back to looking for BUFFER nodes in the graph *)
    let uops = Uop.toposort1 t.uop in
    List.find_map (fun (u : Uop.t) ->
      if u.op = Ops.BUFFER then Schedule.get_realized u.id
      else None
    ) uops

(** Convert to flat float list (triggers realize if needed) *)
let to_float_list (t : t) =
  let _t = realize t in
  match find_realized_buffer t with
  | Some buf ->
    let data = Device.copyout_floats buf in
    Array.to_list data
  | None ->
    failwith (Printf.sprintf
      "Tensor.to_float_list: no realized buffer found for Tensor(shape=[%s], device=%s)"
      (String.concat "," (List.map string_of_int t.shape)) t.device)

(** OCaml operator overloads *)
let ( + ) = add
let ( - ) = sub
let ( * ) = mul
let ( / ) = div
let ( ~- ) = neg_

(** Debug: print tensor info *)
let to_string (t : t) =
  Printf.sprintf "Tensor(shape=[%s], dtype=%s, device=%s)"
    (String.concat ", " (List.map string_of_int t.shape))
    (Dtype.to_string t.dtype)
    t.device

(** Number of elements *)
let numel (t : t) = Helpers.prod t.shape

(** Number of dimensions *)
let ndim (t : t) = List.length t.shape

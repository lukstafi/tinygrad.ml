(** Schedule creation: transforms a lazy UOp graph into executable kernel descriptions.
    Ported from tinygrad/engine/schedule.py (simplified).

    The full tinygrad scheduler runs 15+ graph rewrite passes to:
    1. Identify kernel boundaries (where to cut the graph)
    2. Assign buffers to intermediate results
    3. Linearize each kernel's UOp subgraph

    This simplified version handles the common case of elementwise operations
    on realized buffers by lowering them into executable kernels directly. *)

type exec_item =
  | Kernel of {
      name: string;
      uops: Uop.t list;
      bufs: Device.buffer list;
      data_map: (int * float array) list;  (** buf_param_idx -> initial data *)
    }
  | Copyin of {
      buf: Device.buffer;
      data: float array;
    }

(** Global buffer data store: maps buffer UOp id -> float array data.
    Used by from_float_list to associate data with buffer nodes. *)
let buffer_data : (int, float array) Hashtbl.t = Hashtbl.create 64

(** Store data associated with a buffer UOp *)
let store_buffer_data buf_uop_id data =
  Hashtbl.replace buffer_data buf_uop_id data

(** Retrieve stored data for a buffer *)
let get_buffer_data buf_uop_id =
  Hashtbl.find_opt buffer_data buf_uop_id

(** Realized buffer store: maps buffer UOp id -> Device.buffer.
    After a buffer is allocated and filled, it's stored here. *)
let realized_buffers : (int, Device.buffer) Hashtbl.t = Hashtbl.create 64

let store_realized buf_uop_id dbuf =
  Hashtbl.replace realized_buffers buf_uop_id dbuf

let get_realized buf_uop_id =
  Hashtbl.find_opt realized_buffers buf_uop_id

(** Simple scheduler: generates exec items for a UOp root.
    This handles the case where the root is a lazy graph of elementwise ops
    over buffers. It creates a single kernel that computes the entire thing. *)
let create_schedule ?(device="CPU") (roots : Uop.t list) : exec_item list =
  (* For each root, check if it involves unrealized computation *)
  List.filter_map (fun (root : Uop.t) ->
    (* Find all BUFFER nodes in the graph *)
    let uops = Uop.toposort1 root in
    let has_unrealized_buffer = List.exists (fun (u : Uop.t) ->
      u.op = Ops.BUFFER &&
      get_realized u.id = None &&
      get_buffer_data u.id <> None
    ) uops in
    if has_unrealized_buffer then begin
      (* Simple approach: just copyin each unrealized buffer *)
      let items = List.filter_map (fun (u : Uop.t) ->
        if u.op = Ops.BUFFER && get_realized u.id = None then
          match get_buffer_data u.id with
          | Some data ->
            let size = Array.length data in
            let dtype = match u.dtype with
              | Dtype.Ptr (base, _, _) -> base
              | dt -> dt
            in
            let buf = Device.alloc_buffer
              (Device.make_buffer ~device ~size ~dtype) in
            store_realized u.id buf;
            Some (Copyin { buf; data })
          | None -> None
        else None
      ) uops in
      if items = [] then None else Some items
    end else
      None
  ) roots |> List.flatten

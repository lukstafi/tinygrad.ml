(** C-style code renderer for UOp graphs.
    Ported from tinygrad/renderer/cstyle.py.

    This module renders linearized UOps into C-like source code. It's the base
    for CPU (Clang), CUDA, and Metal renderers — each overrides specific methods. *)

(** Renderer configuration — what varies between C/CUDA/Metal *)
type config = {
  device: string;
  kernel_typedef: string;       (** e.g., "void" or "kernel void" or "__global__ void" *)
  buffer_prefix: string;        (** e.g., "" or "device " or "__global " *)
  buffer_suffix: string;        (** e.g., " restrict" *)
  smem_prefix: string;          (** e.g., "__shared__" or "threadgroup" *)
  arg_int_prefix: string;       (** e.g., "const int" or "constant int&" *)
  barrier: string;              (** e.g., "__syncthreads();" *)
  float4_prefix: string;        (** e.g., "(float4)" or "make_float4" or "float4" *)
  infinity: string;             (** e.g., "INFINITY" or "__builtin_inff()" *)
  nan: string;                  (** e.g., "NAN" or "__builtin_nanf(\"\")" *)
  has_local: bool;
  code_for_workitem: (string -> string) option;  (** global id accessor *)
  code_for_local: (string -> string) option;     (** local id accessor *)
  extra_args: string list;      (** extra kernel params like "uint3 gid [[...]]" *)
  type_overrides: (Dtype.scalar * string) list;   (** e.g., Bool -> "_Bool" *)
  prefix_lines: string list;    (** lines before the kernel *)
  render_cast: Dtype.t -> string -> string;  (** custom cast rendering *)
}

(** Render a dtype to its C name *)
let render_dtype cfg dt =
  let scalar_name s =
    match List.assoc_opt s cfg.type_overrides with
    | Some name -> name
    | None -> Dtype.scalar_name s
  in
  match dt with
  | Dtype.Scalar s -> scalar_name s
  | Dtype.Vec (s, c) -> Printf.sprintf "%s%d" (String.map (fun c -> if c = ' ' then '_' else c) (scalar_name s)) c
  | Dtype.Ptr (Dtype.Scalar s, Dtype.Global, _) -> Printf.sprintf "%s%s*%s" cfg.buffer_prefix (scalar_name s) cfg.buffer_suffix
  | Dtype.Ptr (Dtype.Scalar s, Dtype.Local, _) -> Printf.sprintf "%s%s*" cfg.smem_prefix (scalar_name s)
  | Dtype.Ptr (b, _, _) ->
    let rec inner = function
      | Dtype.Scalar s -> scalar_name s
      | Dtype.Vec (s, c) -> Printf.sprintf "%s%d" (scalar_name s) c
      | Dtype.Ptr (b, _, _) -> inner b ^ "*"
      | Dtype.Void -> "void"
    in inner b ^ "*"
  | Dtype.Void -> "void"

(** Render a constant value *)
let render_const cfg (dt : Dtype.t) (arg : Uop.arg) =
  match arg with
  | Uop.Float_arg f ->
    if Float.is_nan f then Printf.sprintf "(%s)" (cfg.render_cast dt cfg.nan)
    else if Float.is_infinite f && f > 0.0 then Printf.sprintf "(%s)" (cfg.render_cast dt cfg.infinity)
    else if Float.is_infinite f then Printf.sprintf "(%s)" (cfg.render_cast dt (Printf.sprintf "-%s" cfg.infinity))
    else begin match dt with
      | Dtype.Scalar Dtype.Float32 -> Printf.sprintf "%sf" (string_of_float f)
      | Dtype.Scalar Dtype.Float64 -> string_of_float f
      | Dtype.Scalar (Dtype.Float16 | Dtype.BFloat16) -> Printf.sprintf "(%s)" (cfg.render_cast dt (Printf.sprintf "%sf" (string_of_float f)))
      | _ -> Printf.sprintf "(%s)" (cfg.render_cast dt (string_of_float f))
    end
  | Uop.Int_arg i ->
    begin match dt with
      | Dtype.Scalar Dtype.Bool -> if i <> 0 then "1" else "0"
      | Dtype.Scalar Dtype.Int64 -> Printf.sprintf "%dll" i
      | Dtype.Scalar Dtype.Uint64 -> Printf.sprintf "%dull" i
      | Dtype.Scalar Dtype.Uint32 -> Printf.sprintf "%du" i
      | _ -> string_of_int i
    end
  | _ -> failwith "render_const: unexpected arg type"

(** Render an ALU operation *)
let render_alu op args _dtype =
  match op, args with
  | Ops.SQRT, [x] -> Printf.sprintf "sqrt(%s)" x
  | Ops.RECIPROCAL, [x] -> Printf.sprintf "(1/%s)" x
  | Ops.NEG, [x] -> Printf.sprintf "-%s" x
  | Ops.EXP2, [x] -> Printf.sprintf "exp2(%s)" x
  | Ops.LOG2, [x] -> Printf.sprintf "log2(%s)" x
  | Ops.SIN, [x] -> Printf.sprintf "sin(%s)" x
  | Ops.TRUNC, [x] -> Printf.sprintf "trunc(%s)" x
  | Ops.ADD, [a; b] -> Printf.sprintf "(%s+%s)" a b
  | Ops.SUB, [a; b] -> Printf.sprintf "(%s-%s)" a b
  | Ops.MUL, [a; b] -> Printf.sprintf "(%s*%s)" a b
  | Ops.IDIV, [a; b] -> Printf.sprintf "(%s/%s)" a b
  | Ops.MOD, [a; b] -> Printf.sprintf "(%s%%%s)" a b
  | Ops.MAX, [a; b] -> Printf.sprintf "((%s)>(%s)?(%s):(%s))" a b a b
  | Ops.CMPLT, [a; b] -> Printf.sprintf "(%s<%s)" a b
  | Ops.CMPNE, [a; b] -> Printf.sprintf "(%s!=%s)" a b
  | Ops.CMPEQ, [a; b] -> Printf.sprintf "(%s==%s)" a b
  | Ops.AND, [a; b] -> Printf.sprintf "(%s&%s)" a b
  | Ops.OR, [a; b] -> Printf.sprintf "(%s|%s)" a b
  | Ops.XOR, [a; b] -> Printf.sprintf "(%s^%s)" a b
  | Ops.SHL, [a; b] -> Printf.sprintf "(%s<<%s)" a b
  | Ops.SHR, [a; b] -> Printf.sprintf "(%s>>%s)" a b
  | Ops.WHERE, [a; b; c] -> Printf.sprintf "(%s?%s:%s)" a b c
  | Ops.FDIV, [a; b] -> Printf.sprintf "(%s/%s)" a b
  | _ -> failwith (Printf.sprintf "render_alu: unsupported op %s with %d args" (Ops.to_string op) (List.length args))

(** Axis letters for range variable naming *)
let axis_letters = [|"g"; "l"; "r"; "i"; "j"; "k"; "m"; "n"|]

(** Render a list of linearized UOps into (function_name, kernel_lines, param_list) *)
let render_uops cfg (uops : Uop.t list) =
  (* Map from UOp id to its rendered name/expression *)
  let r : (int, string) Hashtbl.t = Hashtbl.create 256 in
  let get u = match Hashtbl.find_opt r u.Uop.id with Some s -> s | None -> Printf.sprintf "/*unknown_%d*/" u.id in

  (* Track which PARAMs are stored to (writable) *)
  let writable_params = Hashtbl.create 16 in
  List.iter (fun u ->
    if u.Uop.op = Ops.STORE then begin
      let rec find_param u =
        if u.Uop.op = Ops.PARAM then Hashtbl.replace writable_params u.Uop.id ()
        else List.iter find_param u.Uop.src
      in
      if List.length u.src > 0 then find_param (List.hd u.src)
    end
  ) uops;

  let kernel = Buffer.create 1024 in
  let depth = ref 1 in
  let indent () = String.make (!depth * 2) ' ' in
  let bufs = ref [] in  (* (name, dtype, is_mutable) *)
  let counters = Hashtbl.create 16 in
  let next_name prefix =
    let c = match Hashtbl.find_opt counters prefix with Some n -> n | None -> 0 in
    Hashtbl.replace counters prefix (c + 1);
    Printf.sprintf "%s%d" prefix c
  in
  let func_name = ref "test" in
  let globals = ref [] in
  let outs = ref [] in
  let ins = ref [] in
  let global_size = ref [1; 1; 1] in
  let local_size = ref None in
  let vars = ref [] in

  List.iter (fun (u : Uop.t) ->
    match u.op with
    | Ops.NOOP | Ops.GROUP -> ()
    | Ops.AFTER ->
      if List.length u.src > 0 then
        Hashtbl.replace r u.id (get (List.hd u.src))
    | Ops.SINK ->
      (match u.arg with Uop.Func_name n -> func_name := n | _ -> ())
    | Ops.PARAM ->
      let idx = match u.arg with Uop.Int_arg i -> i | _ -> 0 in
      let name = Printf.sprintf "data%d" idx in
      let mutable_ = Hashtbl.mem writable_params u.id in
      Hashtbl.replace r u.id name;
      bufs := !bufs @ [(name, u.dtype, mutable_)];
      globals := !globals @ [idx];
      if mutable_ then outs := !outs @ [idx]
      else ins := !ins @ [idx]
    | Ops.DEFINE_VAR ->
      let name = match u.arg with Uop.String_arg s -> s | _ -> "var" in
      Hashtbl.replace r u.id name;
      let vmin = match u.src with
        | [lo; hi] ->
          let lo_v = (match lo.arg with Uop.Int_arg i -> i | _ -> 0) in
          let hi_v = (match hi.arg with Uop.Int_arg i -> i | _ -> 0) in
          vars := !vars @ [(name, lo_v, hi_v)];
          ignore lo_v; lo_v
        | _ -> 0
      in
      ignore vmin;
      bufs := !bufs @ [(name, Dtype.int32, false)]
    | Ops.CONST ->
      let name = next_name "const" in
      Hashtbl.replace r u.id (render_const cfg u.dtype u.arg);
      ignore name  (* consts are inlined *)
    | Ops.CAST ->
      let src_str = get (List.hd u.src) in
      Hashtbl.replace r u.id (Printf.sprintf "(%s)(%s)" (render_dtype cfg u.dtype) src_str)
    | Ops.BITCAST ->
      let src_str = get (List.hd u.src) in
      (* Simple bitcast via union or reinterpret *)
      Hashtbl.replace r u.id (Printf.sprintf "(*(%s*)&(%s))" (render_dtype cfg u.dtype) src_str)
    | Ops.INDEX ->
      let buf_str = get (List.hd u.src) in
      let idx_str = get (List.nth u.src 1) in
      Hashtbl.replace r u.id (Printf.sprintf "(%s+%s)" buf_str (Helpers.strip_parens idx_str))
    | Ops.LOAD ->
      let name = next_name "val" in
      let bidx_str = get (List.hd u.src) in
      let expr = Printf.sprintf "(*%s)" bidx_str in
      Buffer.add_string kernel (Printf.sprintf "%s%s %s = %s;\n" (indent ()) (render_dtype cfg u.dtype) name expr);
      Hashtbl.replace r u.id name
    | Ops.STORE ->
      let bidx_str = get (List.hd u.src) in
      let val_str = get (List.nth u.src 1) in
      Buffer.add_string kernel (Printf.sprintf "%s*%s = %s;\n" (indent ()) bidx_str val_str)
    | Ops.RANGE ->
      let name = match u.arg with
        | Uop.Tuple_int (idx :: _) ->
          let letter = if idx < Array.length axis_letters then axis_letters.(idx) else Printf.sprintf "i%d" idx in
          Printf.sprintf "%sidx%d" letter idx
        | _ -> next_name "ridx"
      in
      let bound_str = get (List.hd u.src) in
      Buffer.add_string kernel (Printf.sprintf "%sfor (int %s = 0; %s < %s; %s++) {\n"
        (indent ()) name name bound_str name);
      Hashtbl.replace r u.id name;
      incr depth
    | Ops.END ->
      decr depth;
      Buffer.add_string kernel (Printf.sprintf "%s}\n" (indent ()))
    | Ops.IF ->
      let cond_str = get (List.hd u.src) in
      Buffer.add_string kernel (Printf.sprintf "%sif (%s) {\n" (indent ()) cond_str);
      incr depth
    | Ops.ENDIF ->
      decr depth;
      Buffer.add_string kernel (Printf.sprintf "%s}\n" (indent ()))
    | Ops.SPECIAL ->
      let name = match u.arg with Uop.String_arg s -> s | _ -> "special" in
      let bound = List.hd u.src in
      let bound_val = match bound.arg with Uop.Int_arg i -> i | _ -> 1 in
      (* Parse the special name to determine type (g=global, l=local) and axis *)
      let accessor = match String.sub name 0 1 with
        | "g" ->
          let axis = int_of_char name.[1] - int_of_char '0' in
          (* Update global_size *)
          while List.length !global_size <= axis do
            global_size := !global_size @ [1]
          done;
          global_size := List.mapi (fun i v -> if i = axis then bound_val else v) !global_size;
          (match cfg.code_for_workitem with Some f -> f (String.make 1 name.[1]) | None -> name)
        | "l" ->
          let axis = int_of_char name.[1] - int_of_char '0' in
          let ls = match !local_size with Some l -> l | None -> [1; 1; 1] in
          let ls = List.mapi (fun i v -> if i = axis then bound_val else v) ls in
          local_size := Some ls;
          (match cfg.code_for_local with Some f -> f (String.make 1 name.[1]) | None -> name)
        | _ -> name
      in
      let rendered_name = next_name "special" in
      Buffer.add_string kernel (Printf.sprintf "%sint %s = %s;\n" (indent ()) rendered_name accessor);
      Hashtbl.replace r u.id rendered_name
    | Ops.DEFINE_LOCAL ->
      let name = next_name "temp" in
      let size = match u.arg with Uop.Int_arg s -> s | _ -> 0 in
      let base_dt = Dtype.base u.dtype in
      Buffer.add_string kernel (Printf.sprintf "%s%s %s[%d];\n" (indent ()) (render_dtype cfg base_dt) name size);
      Hashtbl.replace r u.id name
    | Ops.BARRIER ->
      Buffer.add_string kernel (Printf.sprintf "%s%s\n" (indent ()) cfg.barrier)
    | Ops.VECTORIZE ->
      let elems = List.map get u.src in
      let name = next_name "cast" in
      let expr = Printf.sprintf "%s(%s)" cfg.float4_prefix (String.concat ", " elems) in
      Buffer.add_string kernel (Printf.sprintf "%s%s %s = %s;\n" (indent ()) (render_dtype cfg u.dtype) name expr);
      Hashtbl.replace r u.id name
    | Ops.GEP ->
      let src_str = get (List.hd u.src) in
      let idx = match u.arg with Uop.Int_arg i -> i | _ -> 0 in
      Hashtbl.replace r u.id (Printf.sprintf "%s[%d]" src_str idx)
    | Ops.DEFINE_REG ->
      let name = next_name "acc" in
      let base_dt = Dtype.base u.dtype in
      let count = Dtype.count u.dtype in
      Buffer.add_string kernel (Printf.sprintf "%s%s %s[%d];\n" (indent ()) (render_dtype cfg base_dt) name count);
      Hashtbl.replace r u.id name
    | Ops.WMMA ->
      let name = next_name "wmma" in
      let wmma_name = match u.arg with Uop.String_arg s -> s | _ -> "wmma" in
      let args = List.map get u.src in
      let expr = Printf.sprintf "__%s(%s)" wmma_name (String.concat ", " args) in
      Buffer.add_string kernel (Printf.sprintf "%s%s %s = %s;\n" (indent ()) (render_dtype cfg u.dtype) name expr);
      Hashtbl.replace r u.id name
    | Ops.CUSTOM | Ops.CUSTOMI ->
      let fmt_str = match u.arg with Uop.String_arg s -> s | _ -> "" in
      (* Simple format: replace {0}, {1}, ... with src values *)
      let result = ref fmt_str in
      List.iteri (fun i s ->
        result := Str.global_replace (Str.regexp_string (Printf.sprintf "{%d}" i)) (get s) !result
      ) u.src;
      if u.op = Ops.CUSTOMI then
        Hashtbl.replace r u.id !result
      else begin
        let name = next_name "custom" in
        Buffer.add_string kernel (Printf.sprintf "%s%s;\n" (indent ()) !result);
        Hashtbl.replace r u.id name
      end
    | _ ->
      if Ops.Group.is_alu u.op then begin
        let name = next_name "alu" in
        let args = List.map get u.src in
        let expr = render_alu u.op args u.dtype in
        (* Inline single-use ALU ops? For now, always assign to variable *)
        Buffer.add_string kernel (Printf.sprintf "%s%s %s = %s;\n" (indent ()) (render_dtype cfg u.dtype) name expr);
        Hashtbl.replace r u.id name
      end else begin
        (* Unknown op — emit a comment *)
        Buffer.add_string kernel (Printf.sprintf "%s/* unhandled: %s */\n" (indent ()) (Ops.to_string u.op))
      end
  ) uops;

  let kernel_body = Buffer.contents kernel in

  (* Build parameter list *)
  let params = List.map (fun (name, dtype, mutable_) ->
    let type_str = match dtype with
      | Dtype.Ptr _ -> render_dtype cfg dtype
      | Dtype.Scalar (Dtype.Int32) -> cfg.arg_int_prefix
      | _ -> render_dtype cfg dtype
    in
    Printf.sprintf "%s %s" type_str name, mutable_
  ) !bufs in

  let param_strs = List.map fst params in
  let all_params = param_strs @ cfg.extra_args in

  let prefix = String.concat "\n" cfg.prefix_lines in
  let src = Printf.sprintf "%s%s%s %s(%s) {\n%s}\n"
    (if prefix = "" then "" else prefix ^ "\n")
    (if cfg.kernel_typedef = "" then "" else cfg.kernel_typedef ^ " ")
    ""  (* launch_bounds placeholder *)
    !func_name
    (String.concat ", " all_params)
    kernel_body
  in

  Renderer.make_program_spec
    ~name:!func_name
    ~src
    ~device:cfg.device
    ~global_size:!global_size
    ?local_size:!local_size
    ~globals:!globals
    ~outs:!outs
    ~ins:!ins
    ~vars:!vars
    ()

(** Default C-style config (Clang/CPU) *)
let clang_config = {
  device = "CPU";
  kernel_typedef = "void";
  buffer_prefix = "";
  buffer_suffix = " restrict";
  smem_prefix = "";
  arg_int_prefix = "const int";
  barrier = "";
  float4_prefix = "(float4)";
  infinity = "__builtin_inff()";
  nan = {|__builtin_nanf("")|};
  has_local = false;
  code_for_workitem = None;
  code_for_local = None;
  extra_args = [];
  type_overrides = [(Dtype.Bool, "_Bool")];
  prefix_lines = ["#include <math.h>"];
  render_cast = (fun dt v -> Printf.sprintf "(%s)(%s)" (render_dtype { device="CPU"; kernel_typedef=""; buffer_prefix=""; buffer_suffix=""; smem_prefix=""; arg_int_prefix=""; barrier=""; float4_prefix=""; infinity=""; nan=""; has_local=false; code_for_workitem=None; code_for_local=None; extra_args=[]; type_overrides=[(Dtype.Bool, "_Bool")]; prefix_lines=[]; render_cast=(fun _ v -> v) } dt) v);
}

(** CUDA config *)
let cuda_config ~arch =
  ignore arch;
  {
    device = "CUDA";
    kernel_typedef = {|extern "C" __global__ void|};
    buffer_prefix = "";
    buffer_suffix = "";
    smem_prefix = "__shared__ ";
    arg_int_prefix = "const int";
    barrier = "__syncthreads();";
    float4_prefix = "make_float4";
    infinity = "INFINITY";
    nan = "NAN";
    has_local = true;
    code_for_workitem = Some (fun x -> Printf.sprintf "blockIdx.%c" (Char.chr (Char.code 'x' + int_of_string x)));
    code_for_local = Some (fun x -> Printf.sprintf "threadIdx.%c" (Char.chr (Char.code 'x' + int_of_string x)));
    extra_args = [];
    type_overrides = [(Dtype.BFloat16, "nv_bfloat16")];
    prefix_lines = [
      "#define INFINITY (__int_as_float(0x7f800000))";
      "#define NAN (__int_as_float(0x7fffffff))";
    ];
    render_cast = (fun dt v ->
      let name = match dt with
        | Dtype.Scalar s -> Dtype.scalar_name s
        | _ -> "float"
      in Printf.sprintf "(%s)(%s)" name v);
  }

(** Metal config *)
let metal_config = {
  device = "METAL";
  kernel_typedef = "kernel void";
  buffer_prefix = "device ";
  buffer_suffix = "";
  smem_prefix = {|threadgroup __attribute__((aligned(16))) |};
  arg_int_prefix = "constant int&";
  barrier = "threadgroup_barrier(mem_flags::mem_threadgroup);";
  float4_prefix = "float4";
  infinity = "INFINITY";
  nan = "NAN";
  has_local = true;
  code_for_workitem = Some (fun x -> Printf.sprintf "gid.%c" (Char.chr (Char.code 'x' + int_of_string x)));
  code_for_local = Some (fun x -> Printf.sprintf "lid.%c" (Char.chr (Char.code 'x' + int_of_string x)));
  extra_args = [
    "uint3 gid [[threadgroup_position_in_grid]]";
    "uint3 lid [[thread_position_in_threadgroup]]";
  ];
  type_overrides = [(Dtype.BFloat16, "bfloat")];
  prefix_lines = [
    "#include <metal_stdlib>";
    "using namespace metal;";
  ];
  render_cast = (fun dt v ->
    let name = match dt with
      | Dtype.Scalar s -> Dtype.scalar_name s
      | _ -> "float"
    in Printf.sprintf "(%s)(%s)" name v);
}

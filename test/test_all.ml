(** Tests for tinygrad_ml core modules *)

let test_count = ref 0
let pass_count = ref 0
let fail_count = ref 0

let check name cond =
  incr test_count;
  if cond then begin
    incr pass_count;
    Printf.printf "  PASS: %s\n" name
  end else begin
    incr fail_count;
    Printf.printf "  FAIL: %s\n" name
  end

let section name =
  Printf.printf "\n=== %s ===\n" name

(* ---- DType tests ---- *)
let test_dtype () =
  section "DType";
  check "float32 bitsize" (Dtype.bitsize Dtype.float32 = 32);
  check "float32 itemsize" (Dtype.itemsize Dtype.float32 = 4);
  check "int32 bitsize" (Dtype.bitsize Dtype.int32 = 32);
  check "bool bitsize" (Dtype.bitsize Dtype.bool = 1);
  check "float64 itemsize" (Dtype.itemsize Dtype.float64 = 8);
  check "vec4 bitsize" (Dtype.bitsize (Dtype.vec Dtype.Float32 4) = 128);
  check "vec4 itemsize" (Dtype.itemsize (Dtype.vec Dtype.Float32 4) = 16);
  check "scalar_of vec" (Dtype.scalar_of (Dtype.vec Dtype.Float32 4) = Dtype.Float32);
  check "count scalar" (Dtype.count Dtype.float32 = 1);
  check "count vec4" (Dtype.count (Dtype.vec Dtype.Float32 4) = 4);
  check "is_float Float32" (Dtype.is_float Dtype.Float32);
  check "not is_float Int32" (not (Dtype.is_float Dtype.Int32));
  check "is_int Int32" (Dtype.is_int Dtype.Int32);
  check "is_unsigned Uint8" (Dtype.is_unsigned Dtype.Uint8);
  check "ptr base" (Dtype.base (Dtype.ptr Dtype.float32) = Dtype.float32);
  check "name float32" (Dtype.name Dtype.float32 = "float");
  check "name int32" (Dtype.name Dtype.int32 = "int");
  check "to_string float32" (Dtype.to_string Dtype.float32 = "dtypes.float");
  ()

(* ---- Ops tests ---- *)
let test_ops () =
  section "Ops";
  check "to_string ADD" (Ops.to_string Ops.ADD = "ADD");
  check "to_string CONST" (Ops.to_string Ops.CONST = "CONST");
  check "is_alu ADD" (Ops.Group.is_alu Ops.ADD);
  check "is_alu CONST" (not (Ops.Group.is_alu Ops.CONST));
  check "is_binary ADD" (Ops.Group.is_binary Ops.ADD);
  check "is_unary SIN" (Ops.Group.is_unary Ops.SIN);
  check "is_ternary WHERE" (Ops.Group.is_ternary Ops.WHERE);
  check "is_movement RESHAPE" (Ops.Group.is_movement Ops.RESHAPE);
  check "is_commutative ADD" (Ops.Group.is_commutative Ops.ADD);
  check "not is_commutative SUB" (not (Ops.Group.is_commutative Ops.SUB));
  check "identity ADD" (Ops.identity_element Ops.ADD Dtype.float32 = 0.0);
  check "identity MUL" (Ops.identity_element Ops.MUL Dtype.float32 = 1.0);
  check "compare ADD < CONST" (Ops.compare Ops.ADD Ops.CONST < 0);
  ()

(* ---- UOp tests ---- *)
let test_uop () =
  section "UOp";
  (* Basic construction *)
  let c1 = Uop.const Dtype.float32 1.0 in
  let c2 = Uop.const Dtype.float32 2.0 in
  check "const op" (c1.op = Ops.CONST);
  check "const dtype" (c1.dtype = Dtype.float32);
  check "const arg" (c1.arg = Uop.Float_arg 1.0);

  (* Hash-consing: same args should return same node *)
  let c1_dup = Uop.const Dtype.float32 1.0 in
  check "hash-consing identity" (Uop.equal c1 c1_dup);

  (* Different values should be different nodes *)
  check "different consts differ" (not (Uop.equal c1 c2));

  (* ALU operations *)
  let sum = Uop.add c1 c2 in
  check "add op" (sum.op = Ops.ADD);
  check "add src count" (List.length sum.src = 2);
  check "add src[0]" (Uop.equal (List.hd sum.src) c1);
  check "add src[1]" (Uop.equal (List.nth sum.src 1) c2);

  (* Toposort *)
  let sorted = Uop.toposort1 sum in
  check "toposort length" (List.length sorted = 3);
  check "toposort first is const" ((List.hd sorted).op = Ops.CONST);
  check "toposort last is add" ((List.nth sorted 2).op = Ops.ADD);

  (* More complex graph *)
  let prod = Uop.mul c1 c2 in
  let combined = Uop.add sum prod in
  let sorted2 = Uop.toposort1 combined in
  check "complex toposort" (List.length sorted2 >= 4);
  check "complex toposort last" ((List.nth sorted2 (List.length sorted2 - 1)).op = Ops.ADD);

  (* Movement ops *)
  let reshaped = Uop.reshape c1 [1; 1] in
  check "reshape op" (reshaped.op = Ops.RESHAPE);
  check "reshape arg" (reshaped.arg = Uop.Shape [1; 1]);

  (* Cast *)
  let casted = Uop.cast Dtype.int32 c1 in
  check "cast op" (casted.op = Ops.CAST);
  check "cast dtype" (casted.dtype = Dtype.int32);

  (* Variable *)
  let v = Uop.variable "batch_size" 1 64 in
  check "variable op" (v.op = Ops.DEFINE_VAR);
  check "variable arg" (v.arg = Uop.String_arg "batch_size");

  (* Print for debug *)
  Printf.printf "  UOp debug output:\n";
  Uop.print_uops sorted;
  ()

(* ---- Pattern Matcher tests ---- *)
let test_pattern_matcher () =
  section "Pattern Matcher";

  (* Simple constant folding rule: CONST + CONST -> CONST *)
  let fold_add : unit Pattern_matcher.rule = {
    pattern = Pattern_matcher.pat ~ops:[Ops.ADD]
      ~src:[
        Pattern_matcher.pat ~ops:[Ops.CONST] ~name:"a" ();
        Pattern_matcher.pat ~ops:[Ops.CONST] ~name:"b" ();
      ] ();
    rewrite = (fun () bindings ->
      let a = Pattern_matcher.find "a" bindings in
      let b = Pattern_matcher.find "b" bindings in
      match a.arg, b.arg with
      | Uop.Float_arg fa, Uop.Float_arg fb ->
        Some (Uop.const_float a.dtype (fa +. fb))
      | Uop.Int_arg ia, Uop.Int_arg ib ->
        Some (Uop.const_int a.dtype (ia + ib))
      | _ -> None
    );
  } in

  let pm = Pattern_matcher.create [fold_add] in

  (* Test: 1.0 + 2.0 should fold to 3.0 *)
  let c1 = Uop.const Dtype.float32 1.0 in
  let c2 = Uop.const Dtype.float32 2.0 in
  let sum = Uop.add c1 c2 in
  let result = Pattern_matcher.graph_rewrite pm () sum in
  check "const fold add" (result.op = Ops.CONST);
  check "const fold value" (result.arg = Uop.Float_arg 3.0);

  (* Test: nested constant folding: (1+2) + 3 = 6 *)
  let c3 = Uop.const Dtype.float32 3.0 in
  let nested = Uop.add sum c3 in
  let result2 = Pattern_matcher.graph_rewrite pm () nested in
  check "nested const fold" (result2.op = Ops.CONST);
  check "nested const fold value" (result2.arg = Uop.Float_arg 6.0);

  (* Test: non-matching pattern (MUL should not be folded by ADD rule) *)
  let prod = Uop.mul c1 c2 in
  let result3 = Pattern_matcher.graph_rewrite pm () prod in
  check "non-matching pattern" (result3.op = Ops.MUL);

  (* Test: match with any_pat *)
  let any_unary : unit Pattern_matcher.rule = {
    pattern = Pattern_matcher.pat ~ops:[Ops.NEG] ~name:"x" ();
    rewrite = (fun () bindings ->
      let x = Pattern_matcher.find "x" bindings in
      (* Double negation elimination: handled by matching NEG(NEG(x)) *)
      match x.src with
      | [inner] when inner.op = Ops.NEG -> Some (List.hd inner.src)
      | _ -> None
    );
  } in
  let pm2 = Pattern_matcher.create [any_unary] in
  let neg1 = Uop.neg c1 in
  let neg2 = Uop.neg neg1 in
  let result4 = Pattern_matcher.graph_rewrite pm2 () neg2 in
  check "double neg elimination" (Uop.equal result4 c1);
  ()

(* ---- Renderer tests ---- *)
let test_renderer () =
  section "Renderer (CStyle)";

  (* Build a simple kernel: out[i] = in[i] + 1.0 *)
  let in_param = Uop.param 0 (Dtype.ptr Dtype.float32) in
  let out_param = Uop.param 1 (Dtype.ptr Dtype.float32) in
  let n_param = Uop.param 2 Dtype.int32 in
  ignore n_param;
  let i = Uop.range (Uop.const_int Dtype.int32 1024) [0; 0] in
  let in_idx = Uop.index in_param i in
  let val_ = Uop.load in_idx in
  let one = Uop.const Dtype.float32 1.0 in
  let result = Uop.add val_ one in
  let out_idx = Uop.index out_param i in
  let st = Uop.store out_idx result in
  let end_range = Uop.end_ i in
  let kernel = Uop.sink ~name:"add_one" [st; end_range] in

  (* Linearize and render *)
  let uops = Uop.toposort1 kernel in
  Printf.printf "  Linearized UOps (%d nodes):\n" (List.length uops);
  List.iter (fun u -> Printf.printf "    %s\n" (Uop.pp_uop u)) uops;

  let pspec = Cstyle.render_uops Cstyle.clang_config uops in
  Printf.printf "  Rendered C source:\n%s\n" pspec.src;
  check "kernel name" (pspec.name = "add_one");
  check "has source" (String.length pspec.src > 0);
  let has_for = try ignore (Str.search_forward (Str.regexp_string "for") pspec.src 0); true with Not_found -> false in
  check "source has for loop" has_for;
  let has_kernel_name = try ignore (Str.search_forward (Str.regexp_string "add_one") pspec.src 0); true with Not_found -> false in
  check "source has kernel name" has_kernel_name;

  (* Test CUDA rendering *)
  let cuda_pspec = Cstyle.render_uops (Cstyle.cuda_config ~arch:"sm_80") uops in
  Printf.printf "  Rendered CUDA source:\n%s\n" cuda_pspec.src;
  let has_global = try ignore (Str.search_forward (Str.regexp_string "__global__") cuda_pspec.src 0); true with Not_found -> false in
  check "CUDA has __global__" has_global;

  (* Test Metal rendering *)
  let metal_pspec = Cstyle.render_uops Cstyle.metal_config uops in
  Printf.printf "  Rendered Metal source:\n%s\n" metal_pspec.src;
  let has_metal_stdlib = try ignore (Str.search_forward (Str.regexp_string "metal_stdlib") metal_pspec.src 0); true with Not_found -> false in
  check "Metal has metal_stdlib" has_metal_stdlib;
  ()

(* ---- Tensor tests ---- *)
let test_tensor () =
  section "Tensor";
  let t1 = Tensor.zeros [3; 4] in
  check "zeros shape" (t1.shape = [3; 4]);
  check "zeros dtype" (t1.dtype = Dtype.float32);
  check "zeros device" (t1.device = "CPU");

  let t2 = Tensor.ones [3; 4] in
  check "ones shape" (t2.shape = [3; 4]);

  let t3 = Tensor.full [2; 3] 5.0 in
  check "full shape" (t3.shape = [2; 3]);

  (* Elementwise ops *)
  let t4 = Tensor.add t1 t2 in
  check "add shape" (t4.shape = [3; 4]);
  check "add dtype" (t4.dtype = Dtype.float32);

  let t5 = Tensor.mul t1 t2 in
  check "mul shape" (t5.shape = [3; 4]);

  let t6 = Tensor.neg_ t1 in
  check "neg shape" (t6.shape = [3; 4]);

  (* Movement ops *)
  let t7 = Tensor.reshape t1 [12] in
  check "reshape shape" (t7.shape = [12]);

  let t8 = Tensor.reshape t1 [4; 3] in
  check "reshape shape 2" (t8.shape = [4; 3]);

  (* Operators *)
  let open Tensor in
  let t9 = t1 + t2 in
  check "operator +" (t9.shape = [3; 4]);
  let t10 = t1 * t2 in
  check "operator *" (t10.shape = [3; 4]);
  let t11 = t1 - t2 in
  check "operator -" (t11.shape = [3; 4]);

  (* Reduce *)
  let t12 = Tensor.sum ~axes:[1] t1 in
  check "sum shape" (t12.shape = [3; 1]);

  let t13 = Tensor.mean ~axes:[0] t1 in
  check "mean shape" (t13.shape = [1; 4]);

  (* Cast *)
  let t14 = Tensor.cast Dtype.int32 t1 in
  check "cast dtype" (t14.dtype = Dtype.int32);

  (* to_string *)
  let s = Tensor.to_string t1 in
  check "to_string" (String.length s > 0);
  Printf.printf "  %s\n" s;

  (* numel / ndim *)
  check "numel" (Tensor.numel t1 = 12);
  check "ndim" (Tensor.ndim t1 = 2);
  ()

(* ---- Gradient tests ---- *)
let test_gradient () =
  section "Gradient";
  let x = Uop.const Dtype.float32 3.0 in
  let y = Uop.const Dtype.float32 2.0 in
  let z = Uop.add x y in  (* z = x + y *)

  (* dz/dx = 1, dz/dy = 1 for addition *)
  let grad_z = Uop.const Dtype.float32 1.0 in
  let grads = Gradient.compute_gradient z grad_z [x; y] in
  check "add gradient count" (List.length grads = 2);

  (* Test mul gradient *)
  let w = Uop.mul x y in  (* w = x * y *)
  let grad_w = Uop.const Dtype.float32 1.0 in
  let grads2 = Gradient.compute_gradient w grad_w [x; y] in
  check "mul gradient count" (List.length grads2 = 2);
  (* dw/dx = y = 2.0, dw/dy = x = 3.0 *)
  (* The gradient UOps should contain the right structure *)
  List.iter (fun (target, grad) ->
    Printf.printf "  grad for %%%d: %s\n" target.Uop.id (Uop.pp_uop grad)
  ) grads2;
  ()

(* ---- Helpers tests ---- *)
let test_helpers () =
  section "Helpers";
  check "prod empty" (Helpers.prod [] = 1);
  check "prod [2;3;4]" (Helpers.prod [2; 3; 4] = 24);
  check "ceildiv 7 3" (Helpers.ceildiv 7 3 = 3);
  check "ceildiv 6 3" (Helpers.ceildiv 6 3 = 2);
  check "round_up 7 4" (Helpers.round_up 7 4 = 8);
  check "dedup" (Helpers.dedup [1; 2; 1; 3; 2] = [1; 2; 3]);
  check "partition" (Helpers.partition (fun x -> x > 0) [-1; 2; -3; 4] = ([2; 4], [-1; -3]));
  check "argsort" (Helpers.argsort [3; 1; 2] = [1; 2; 0]);
  check "to_function_name" (Helpers.to_function_name "hello-world.test" = "hello_world_test");
  check "strip_parens" (Helpers.strip_parens "(hello)" = "hello");
  check "strip_parens no" (Helpers.strip_parens "hello" = "hello");
  ()

(* ---- Main ---- *)
let () =
  Printf.printf "tinygrad_ml test suite\n";
  Printf.printf "=====================\n";
  test_helpers ();
  test_dtype ();
  test_ops ();
  test_uop ();
  test_pattern_matcher ();
  test_renderer ();
  test_tensor ();
  test_gradient ();
  Printf.printf "\n=====================\n";
  Printf.printf "Results: %d passed, %d failed, %d total\n" !pass_count !fail_count !test_count;
  if !fail_count > 0 then exit 1

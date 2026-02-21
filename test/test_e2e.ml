(** End-to-end execution tests for tinygrad_ml.
    Tests the full pipeline: UOp IR → render C → compile → execute → verify *)

let pass_count = ref 0
let fail_count = ref 0

let check name cond =
  if cond then (incr pass_count; Printf.printf "  PASS: %s\n%!" name)
  else (incr fail_count; Printf.printf "  FAIL: %s\n%!" name)

let check_float name got expected tolerance =
  let ok = Float.abs (got -. expected) <= tolerance in
  if not ok then Printf.printf "    got %f, expected %f\n%!" got expected;
  check name ok

(** Helper: build, compile, and run a kernel on CPU.
    [bufs] is indexed by param number (param 0 = bufs[0], param 1 = bufs[1], etc.)
    The renderer may reorder params in the function signature; we use pspec.globals
    to map from function argument order to the caller's param indices. *)
let run_cpu_kernel kernel_name uops bufs =
  let pspec = Cstyle.render_uops Cstyle.clang_config uops in
  let module B = (val Device.get_backend "CPU" : Device.Backend) in
  let so_path = B.compile kernel_name pspec.src in
  (* pspec.globals gives param indices in function signature order *)
  let ordered_ptrs = List.map (fun idx ->
    let b = List.nth bufs idx in
    (b : Device.buffer).ptr
  ) pspec.globals in
  B.exec kernel_name so_path ordered_ptrs [];
  pspec

(* ---- Test 1: Vector Add ---- *)
let test_vector_add () =
  Printf.printf "\n=== Vector Add ===\n%!";
  let n = 256 in
  (* Kernel: out[i] = a[i] + b[i] *)
  let a_param = Uop.param 0 (Dtype.ptr Dtype.float32) in
  let b_param = Uop.param 1 (Dtype.ptr Dtype.float32) in
  let out_param = Uop.param 2 (Dtype.ptr Dtype.float32) in
  let i = Uop.range (Uop.const_int Dtype.int32 n) [0; 0] in
  let a_val = Uop.load (Uop.index a_param i) in
  let b_val = Uop.load (Uop.index b_param i) in
  let sum = Uop.add a_val b_val in
  let st = Uop.store (Uop.index out_param i) sum in
  let end_r = Uop.end_ i in
  let kernel = Uop.sink ~name:"vec_add" [st; end_r] in
  let uops = Uop.toposort1 kernel in

  (* Allocate buffers *)
  let a_buf = Device.alloc_buffer (Device.make_buffer ~device:"CPU" ~size:n ~dtype:Dtype.float32) in
  let b_buf = Device.alloc_buffer (Device.make_buffer ~device:"CPU" ~size:n ~dtype:Dtype.float32) in
  let out_buf = Device.alloc_buffer (Device.make_buffer ~device:"CPU" ~size:n ~dtype:Dtype.float32) in

  (* Fill: a[i] = i, b[i] = 100-i *)
  Device.copyin_floats a_buf (Array.init n Float.of_int);
  Device.copyin_floats b_buf (Array.init n (fun i -> 100.0 -. Float.of_int i));

  (* Execute — bufs indexed by param number: param0=a, param1=b, param2=out *)
  let _pspec = run_cpu_kernel "vec_add" uops [a_buf; b_buf; out_buf] in

  (* Verify: out[i] = i + (100-i) = 100.0 *)
  let out = Device.copyout_floats out_buf in
  let all_correct = Array.for_all (fun v -> Float.abs (v -. 100.0) < 1e-6) out in
  check "vec_add all correct" all_correct;
  check_float "vec_add[0]" out.(0) 100.0 1e-6;
  check_float "vec_add[255]" out.(255) 100.0 1e-6

(* ---- Test 2: Elementwise chain: out = (a + b) * c ---- *)
let test_fused_ops () =
  Printf.printf "\n=== Fused Ops: (a+b)*c ===\n%!";
  let n = 128 in
  let a_param = Uop.param 0 (Dtype.ptr Dtype.float32) in
  let b_param = Uop.param 1 (Dtype.ptr Dtype.float32) in
  let c_param = Uop.param 2 (Dtype.ptr Dtype.float32) in
  let out_param = Uop.param 3 (Dtype.ptr Dtype.float32) in
  let i = Uop.range (Uop.const_int Dtype.int32 n) [0; 0] in
  let a_val = Uop.load (Uop.index a_param i) in
  let b_val = Uop.load (Uop.index b_param i) in
  let c_val = Uop.load (Uop.index c_param i) in
  let sum = Uop.add a_val b_val in
  let prod = Uop.mul sum c_val in
  let st = Uop.store (Uop.index out_param i) prod in
  let end_r = Uop.end_ i in
  let kernel = Uop.sink ~name:"fused_add_mul" [st; end_r] in
  let uops = Uop.toposort1 kernel in

  let a_buf = Device.alloc_buffer (Device.make_buffer ~device:"CPU" ~size:n ~dtype:Dtype.float32) in
  let b_buf = Device.alloc_buffer (Device.make_buffer ~device:"CPU" ~size:n ~dtype:Dtype.float32) in
  let c_buf = Device.alloc_buffer (Device.make_buffer ~device:"CPU" ~size:n ~dtype:Dtype.float32) in
  let out_buf = Device.alloc_buffer (Device.make_buffer ~device:"CPU" ~size:n ~dtype:Dtype.float32) in

  (* a=2, b=3, c=4 → (2+3)*4 = 20 *)
  Device.copyin_floats a_buf (Array.make n 2.0);
  Device.copyin_floats b_buf (Array.make n 3.0);
  Device.copyin_floats c_buf (Array.make n 4.0);

  (* Execute with 4 buffers — bufs indexed by param number *)
  let _pspec = run_cpu_kernel "fused_add_mul" uops [a_buf; b_buf; c_buf; out_buf] in

  let result = Device.copyout_floats out_buf in
  let all_correct = Array.for_all (fun v -> Float.abs (v -. 20.0) < 1e-6) result in
  check "fused (2+3)*4=20" all_correct;
  check_float "fused[0]" result.(0) 20.0 1e-6;
  check_float "fused[127]" result.(127) 20.0 1e-6

(* ---- Test 3: Reduction (sum) ---- *)
let test_reduction () =
  Printf.printf "\n=== Sum Reduction ===\n%!";
  let n = 256 in
  (* Kernel: out[0] = sum(in[i]) for i in 0..n
     We use a local accumulator variable approach to avoid the
     "load outside loop" problem. The init store sets out[0]=0,
     then inside the loop we read out[0], add in[i], and write back. *)
  let in_param = Uop.param 0 (Dtype.ptr Dtype.float32) in
  let out_param = Uop.param 1 (Dtype.ptr Dtype.float32) in
  let zero_idx = Uop.const_int Dtype.int32 0 in
  let out_idx = Uop.index out_param zero_idx in
  let init_store = Uop.store out_idx (Uop.const Dtype.float32 0.0) in
  (* Loop — build a fresh INDEX inside the loop so the LOAD is inside *)
  let i = Uop.range (Uop.const_int Dtype.int32 n) [0; 0] in
  let in_val = Uop.load (Uop.index in_param i) in
  (* Create a new INDEX node for reading inside the loop, dependent on i
     so that it sorts after RANGE. We use an ADD of 0 to create a data
     dependency on i while keeping the index value at 0. *)
  let loop_out_idx = Uop.index out_param (Uop.sub i i) in
  let cur = Uop.load loop_out_idx in
  let new_val = Uop.add cur in_val in
  let st = Uop.store loop_out_idx new_val in
  let end_r = Uop.end_ i in
  let kernel = Uop.sink ~name:"reduce_sum" [init_store; st; end_r] in
  let uops = Uop.toposort1 kernel in

  let in_buf = Device.alloc_buffer (Device.make_buffer ~device:"CPU" ~size:n ~dtype:Dtype.float32) in
  let out_buf = Device.alloc_buffer (Device.make_buffer ~device:"CPU" ~size:1 ~dtype:Dtype.float32) in

  (* Fill with 1.0 → sum should be 256.0 *)
  Device.copyin_floats in_buf (Array.make n 1.0);
  Device.copyin_floats out_buf (Array.make 1 0.0);

  let _pspec = run_cpu_kernel "reduce_sum" uops [in_buf; out_buf] in

  let result = Device.copyout_floats out_buf in
  check_float "reduce_sum" result.(0) (Float.of_int n) 1e-4;

  (* Also test with i values: sum(i for i in 0..n) = n*(n-1)/2 *)
  Device.copyin_floats in_buf (Array.init n Float.of_int);
  Device.copyin_floats out_buf (Array.make 1 0.0);
  let _pspec = run_cpu_kernel "reduce_sum" uops [in_buf; out_buf] in
  let result2 = Device.copyout_floats out_buf in
  let expected = Float.of_int (n * (n - 1) / 2) in
  check_float "reduce_sum(0..255)" result2.(0) expected 1e-1

(* ---- Test 4: Unary ops ---- *)
let test_unary_ops () =
  Printf.printf "\n=== Unary Ops ===\n%!";
  let n = 64 in
  (* Kernel: out[i] = sqrt(in[i]) *)
  let in_param = Uop.param 0 (Dtype.ptr Dtype.float32) in
  let out_param = Uop.param 1 (Dtype.ptr Dtype.float32) in
  let i = Uop.range (Uop.const_int Dtype.int32 n) [0; 0] in
  let in_val = Uop.load (Uop.index in_param i) in
  let result = Uop.sqrt_ in_val in
  let st = Uop.store (Uop.index out_param i) result in
  let end_r = Uop.end_ i in
  let kernel = Uop.sink ~name:"sqrt_kernel" [st; end_r] in
  let uops = Uop.toposort1 kernel in

  let in_buf = Device.alloc_buffer (Device.make_buffer ~device:"CPU" ~size:n ~dtype:Dtype.float32) in
  let out_buf = Device.alloc_buffer (Device.make_buffer ~device:"CPU" ~size:n ~dtype:Dtype.float32) in

  Device.copyin_floats in_buf (Array.init n (fun i -> Float.of_int (i * i)));
  let _pspec = run_cpu_kernel "sqrt_kernel" uops [in_buf; out_buf] in
  let result = Device.copyout_floats out_buf in

  check_float "sqrt(0)" result.(0) 0.0 1e-6;
  check_float "sqrt(1)" result.(1) 1.0 1e-6;
  check_float "sqrt(4)" result.(2) 2.0 1e-6;
  check_float "sqrt(9)" result.(3) 3.0 1e-5;
  check_float "sqrt(3969)" result.(63) 63.0 1e-3

(* ---- Test 5: Multi-backend rendering ---- *)
let test_multi_backend_render () =
  Printf.printf "\n=== Multi-backend Rendering ===\n%!";
  let n = 1024 in
  let in_param = Uop.param 0 (Dtype.ptr Dtype.float32) in
  let out_param = Uop.param 1 (Dtype.ptr Dtype.float32) in
  let i = Uop.range (Uop.const_int Dtype.int32 n) [0; 0] in
  let in_val = Uop.load (Uop.index in_param i) in
  let result = Uop.mul in_val (Uop.const Dtype.float32 2.0) in
  let st = Uop.store (Uop.index out_param i) result in
  let end_r = Uop.end_ i in
  let kernel = Uop.sink ~name:"scale2x" [st; end_r] in
  let uops = Uop.toposort1 kernel in

  (* CPU *)
  let cpu_pspec = Cstyle.render_uops Cstyle.clang_config uops in
  let has_void = try ignore (Str.search_forward (Str.regexp_string "void") cpu_pspec.src 0); true with Not_found -> false in
  check "CPU has void" has_void;

  (* CUDA *)
  let cuda_pspec = Cstyle.render_uops (Cstyle.cuda_config ~arch:"sm_80") uops in
  let has_global = try ignore (Str.search_forward (Str.regexp_string "__global__") cuda_pspec.src 0); true with Not_found -> false in
  check "CUDA has __global__" has_global;
  let has_extern = try ignore (Str.search_forward (Str.regexp_string {|extern "C"|}) cuda_pspec.src 0); true with Not_found -> false in
  check "CUDA has extern C" has_extern;

  (* Metal *)
  let metal_pspec = Cstyle.render_uops Cstyle.metal_config uops in
  let has_kernel = try ignore (Str.search_forward (Str.regexp_string "kernel void") metal_pspec.src 0); true with Not_found -> false in
  check "Metal has kernel void" has_kernel;
  let has_device = try ignore (Str.search_forward (Str.regexp_string "device float") metal_pspec.src 0); true with Not_found -> false in
  check "Metal has device prefix" has_device;
  let has_gid = try ignore (Str.search_forward (Str.regexp_string "threadgroup_position_in_grid") metal_pspec.src 0); true with Not_found -> false in
  check "Metal has gid attribute" has_gid;
  ()

(* ---- Test 6: Metal GPU execution ---- *)
let test_metal_execution () =
  Printf.printf "\n=== Metal GPU Execution ===\n%!";
  let n = 256 in
  (* Kernel: out[i] = in[i] * 2.0 *)
  let in_param = Uop.param 0 (Dtype.ptr Dtype.float32) in
  let out_param = Uop.param 1 (Dtype.ptr Dtype.float32) in
  let i = Uop.range (Uop.const_int Dtype.int32 n) [0; 0] in
  let in_val = Uop.load (Uop.index in_param i) in
  let result = Uop.mul in_val (Uop.const Dtype.float32 2.0) in
  let st = Uop.store (Uop.index out_param i) result in
  let end_r = Uop.end_ i in
  let kernel = Uop.sink ~name:"scale2x" [st; end_r] in
  let uops = Uop.toposort1 kernel in

  (* Render for Metal *)
  let pspec = Cstyle.render_uops Cstyle.metal_config uops in

  (* Allocate Metal buffers *)
  let in_buf = Device.alloc_buffer
    (Device.make_buffer ~device:"METAL" ~size:n ~dtype:Dtype.float32) in
  let out_buf = Device.alloc_buffer
    (Device.make_buffer ~device:"METAL" ~size:n ~dtype:Dtype.float32) in

  (* Fill input: in[i] = i *)
  Device.copyin_floats in_buf (Array.init n Float.of_int);

  (* Compile and execute *)
  let module B = (val Device.get_backend "METAL" : Device.Backend) in
  let _src = B.compile "scale2x" pspec.src in

  (* Reorder buffers according to pspec.globals *)
  let bufs = [| in_buf; out_buf |] in
  let ordered_ptrs = List.map (fun idx -> bufs.(idx).ptr) pspec.globals in
  B.exec "scale2x" pspec.src ordered_ptrs [];

  (* Verify: out[i] = i * 2 *)
  let result_data = Device.copyout_floats out_buf in
  check_float "metal scale2x[0]" result_data.(0) 0.0 1e-6;
  check_float "metal scale2x[1]" result_data.(1) 2.0 1e-6;
  check_float "metal scale2x[10]" result_data.(10) 20.0 1e-5;
  check_float "metal scale2x[255]" result_data.(255) 510.0 1e-3;
  let all_correct = Array.for_all2 (fun got i ->
    Float.abs (got -. Float.of_int i *. 2.0) < 1e-4
  ) result_data (Array.init n Fun.id) in
  check "metal scale2x all correct" all_correct

(* ---- Test 7: Tensor from_float_list -> realize -> to_float_list round-trip ---- *)
let test_tensor_roundtrip () =
  Printf.printf "\n=== Tensor Data Round-Trip ===\n%!";
  let data = [1.0; 2.0; 3.0; 4.0; 5.0] in
  let t = Tensor.from_float_list [5] data in
  let result = Tensor.to_float_list t in
  check "tensor roundtrip length" (List.length result = 5);
  List.iter2 (fun got expected ->
    check_float (Printf.sprintf "tensor roundtrip %.1f" expected) got expected 1e-6
  ) result data

(* ---- Test 8: Tensor lazy compute via schedule/realize ---- *)
let test_tensor_compute () =
  Printf.printf "\n=== Tensor Lazy Compute ===\n%!";
  (* a + b via Tensor API, realized through schedule → kernel → execute *)
  let a = Tensor.from_float_list [4] [1.0; 2.0; 3.0; 4.0] in
  let b = Tensor.from_float_list [4] [10.0; 20.0; 30.0; 40.0] in
  let c = Tensor.add a b in
  let result = Tensor.to_float_list c in
  check "tensor compute length" (List.length result = 4);
  check_float "tensor a+b[0]" (List.nth result 0) 11.0 1e-6;
  check_float "tensor a+b[1]" (List.nth result 1) 22.0 1e-6;
  check_float "tensor a+b[2]" (List.nth result 2) 33.0 1e-6;
  check_float "tensor a+b[3]" (List.nth result 3) 44.0 1e-6;

  (* Chained: (a + b) * a — c was already realized, so c.uop is now a BUFFER *)
  let d = Tensor.mul c a in
  let result2 = Tensor.to_float_list d in
  check_float "tensor (a+b)*a[0]" (List.nth result2 0) 11.0 1e-6;
  check_float "tensor (a+b)*a[1]" (List.nth result2 1) 44.0 1e-6;
  check_float "tensor (a+b)*a[2]" (List.nth result2 2) 99.0 1e-5;
  check_float "tensor (a+b)*a[3]" (List.nth result2 3) 176.0 1e-4;

  (* Unary: sqrt(a * a) = |a| = a (for positive a) *)
  let e = Tensor.mul a a in
  let f = Tensor.sqrt_ e in
  let result3 = Tensor.to_float_list f in
  check_float "tensor sqrt(a*a)[0]" (List.nth result3 0) 1.0 1e-5;
  check_float "tensor sqrt(a*a)[1]" (List.nth result3 1) 2.0 1e-5;
  check_float "tensor sqrt(a*a)[2]" (List.nth result3 2) 3.0 1e-5;
  check_float "tensor sqrt(a*a)[3]" (List.nth result3 3) 4.0 1e-5

(* ---- Test 9: Tensor compute on Metal GPU ---- *)
let test_tensor_compute_metal () =
  Printf.printf "\n=== Tensor Compute on Metal ===\n%!";
  let a = Tensor.from_float_list ~device:"METAL" [3] [2.0; 4.0; 6.0] in
  let b = Tensor.from_float_list ~device:"METAL" [3] [1.0; 2.0; 3.0] in
  let c = Tensor.add a b in
  let result = Tensor.to_float_list c in
  check "metal tensor length" (List.length result = 3);
  check_float "metal tensor a+b[0]" (List.nth result 0) 3.0 1e-6;
  check_float "metal tensor a+b[1]" (List.nth result 1) 6.0 1e-6;
  check_float "metal tensor a+b[2]" (List.nth result 2) 9.0 1e-6

(* ---- Test 10: Tensor const, neg, reciprocal ---- *)
let test_tensor_unary_ops () =
  Printf.printf "\n=== Tensor Unary Ops ===\n%!";
  let a = Tensor.from_float_list [4] [2.0; 4.0; 5.0; 10.0] in

  (* neg *)
  let n = Tensor.neg_ a in
  let rn = Tensor.to_float_list n in
  check_float "neg[0]" (List.nth rn 0) (-2.0) 1e-6;
  check_float "neg[1]" (List.nth rn 1) (-4.0) 1e-6;

  (* reciprocal *)
  let r = Tensor.reciprocal a in
  let rr = Tensor.to_float_list r in
  check_float "recip[0]" (List.nth rr 0) 0.5 1e-6;
  check_float "recip[1]" (List.nth rr 1) 0.25 1e-6;
  check_float "recip[3]" (List.nth rr 3) 0.1 1e-6;

  (* exp2 *)
  let e = Tensor.from_float_list [3] [0.0; 1.0; 3.0] in
  let e2 = Tensor.exp2_ e in
  let re = Tensor.to_float_list e2 in
  check_float "exp2(0)" (List.nth re 0) 1.0 1e-6;
  check_float "exp2(1)" (List.nth re 1) 2.0 1e-6;
  check_float "exp2(3)" (List.nth re 2) 8.0 1e-5

(* ---- Test 11: Tensor div (mul + reciprocal fusion) ---- *)
let test_tensor_div () =
  Printf.printf "\n=== Tensor Div ===\n%!";
  let a = Tensor.from_float_list [4] [10.0; 20.0; 30.0; 40.0] in
  let b = Tensor.from_float_list [4] [2.0; 4.0; 5.0; 8.0] in
  let c = Tensor.div a b in
  let result = Tensor.to_float_list c in
  check_float "div[0]" (List.nth result 0) 5.0 1e-5;
  check_float "div[1]" (List.nth result 1) 5.0 1e-5;
  check_float "div[2]" (List.nth result 2) 6.0 1e-5;
  check_float "div[3]" (List.nth result 3) 5.0 1e-5

(* ---- Test 12: Deep chained compute (3 stages) ---- *)
let test_tensor_deep_chain () =
  Printf.printf "\n=== Tensor Deep Chain ===\n%!";
  (* a=1,2,3 b=10,20,30 *)
  (* c = a+b = 11,22,33 — realize *)
  (* d = c*a = 11,44,99 — realize *)
  (* e = d-b = 1,24,69 *)
  let a = Tensor.from_float_list [3] [1.0; 2.0; 3.0] in
  let b = Tensor.from_float_list [3] [10.0; 20.0; 30.0] in
  let c = Tensor.add a b in
  let _ = Tensor.to_float_list c in  (* force realize of c *)
  let d = Tensor.mul c a in
  let _ = Tensor.to_float_list d in  (* force realize of d *)
  let e = Tensor.sub d b in
  let result = Tensor.to_float_list e in
  check_float "deep chain[0] (c-b)" (List.nth result 0) 1.0 1e-5;
  check_float "deep chain[1]" (List.nth result 1) 24.0 1e-5;
  check_float "deep chain[2]" (List.nth result 2) 69.0 1e-4

(* ---- Test 13: Metal chained compute ---- *)
let test_metal_chain () =
  Printf.printf "\n=== Metal Chained Compute ===\n%!";
  let a = Tensor.from_float_list ~device:"METAL" [3] [3.0; 6.0; 9.0] in
  let b = Tensor.from_float_list ~device:"METAL" [3] [1.0; 2.0; 3.0] in
  let c = Tensor.add a b in
  let _ = Tensor.to_float_list c in
  let d = Tensor.mul c b in
  let result = Tensor.to_float_list d in
  check_float "metal chain (a+b)*b[0]" (List.nth result 0) 4.0 1e-5;   (* (3+1)*1 = 4 *)
  check_float "metal chain (a+b)*b[1]" (List.nth result 1) 16.0 1e-5;  (* (6+2)*2 = 16 *)
  check_float "metal chain (a+b)*b[2]" (List.nth result 2) 36.0 1e-4   (* (9+3)*3 = 36 *)

(* ---- Test 14: Tensor const_like and full ---- *)
let test_tensor_const () =
  Printf.printf "\n=== Tensor Const ===\n%!";
  let t = Tensor.full [3] 7.0 in
  let result = Tensor.to_float_list t in
  check "const length" (List.length result = 3);
  check_float "full(7)[0]" (List.nth result 0) 7.0 1e-6;
  check_float "full(7)[1]" (List.nth result 1) 7.0 1e-6;
  check_float "full(7)[2]" (List.nth result 2) 7.0 1e-6;

  (* zeros/ones *)
  let z = Tensor.zeros [2] in
  let rz = Tensor.to_float_list z in
  check_float "zeros[0]" (List.nth rz 0) 0.0 1e-6;
  check_float "zeros[1]" (List.nth rz 1) 0.0 1e-6;

  let o = Tensor.ones [2] in
  let ro = Tensor.to_float_list o in
  check_float "ones[0]" (List.nth ro 0) 1.0 1e-6;
  check_float "ones[1]" (List.nth ro 1) 1.0 1e-6

(* ---- Test 15: Tensor sum reduction ---- *)
let test_tensor_sum () =
  Printf.printf "\n=== Tensor Sum Reduction ===\n%!";
  let a = Tensor.from_float_list [5] [1.0; 2.0; 3.0; 4.0; 5.0] in
  let s = Tensor.sum a in
  let result = Tensor.to_float_list s in
  check "sum length" (List.length result = 1);
  check_float "sum([1..5])" (List.nth result 0) 15.0 1e-4;

  (* Sum of computed expression: sum(a + b) *)
  let b = Tensor.from_float_list [4] [10.0; 20.0; 30.0; 40.0] in
  let c = Tensor.from_float_list [4] [1.0; 2.0; 3.0; 4.0] in
  let d = Tensor.add b c in
  let s2 = Tensor.sum d in
  let result2 = Tensor.to_float_list s2 in
  check_float "sum(b+c)" (List.nth result2 0) 110.0 1e-4

(* ---- Test 16: Tensor max reduction ---- *)
let test_tensor_max () =
  Printf.printf "\n=== Tensor Max Reduction ===\n%!";
  let a = Tensor.from_float_list [5] [3.0; 1.0; 5.0; 2.0; 4.0] in
  let m = Tensor.max_ a in
  let result = Tensor.to_float_list m in
  check "max length" (List.length result = 1);
  check_float "max([3,1,5,2,4])" (List.nth result 0) 5.0 1e-6

(* ---- Test 17: Tensor mean reduction ---- *)
let test_tensor_mean () =
  Printf.printf "\n=== Tensor Mean Reduction ===\n%!";
  let a = Tensor.from_float_list [4] [2.0; 4.0; 6.0; 8.0] in
  let m = Tensor.mean a in
  let result = Tensor.to_float_list m in
  check "mean length" (List.length result = 1);
  check_float "mean([2,4,6,8])" (List.nth result 0) 5.0 1e-5

(* ---- Test 18: Partial-axis reduction ---- *)
let test_partial_reduction () =
  Printf.printf "\n=== Partial-Axis Reduction ===\n%!";
  Schedule.reset ();
  (* Create a [3;4] tensor and sum along axis 1 → [3;1] *)
  let data = [1.0; 2.0; 3.0; 4.0;   (* row 0: sum=10 *)
              5.0; 6.0; 7.0; 8.0;   (* row 1: sum=26 *)
              9.0; 10.0; 11.0; 12.0] (* row 2: sum=42 *) in
  let a = Tensor.from_float_list [3; 4] data in
  let s = Tensor.sum ~axes:[1] a in
  check "partial sum shape" (s.shape = [3; 1]);
  let result = Tensor.to_float_list s in
  check "partial sum length" (List.length result = 3);
  check_float "partial sum row0" (List.nth result 0) 10.0 1e-5;
  check_float "partial sum row1" (List.nth result 1) 26.0 1e-5;
  check_float "partial sum row2" (List.nth result 2) 42.0 1e-5;

  (* Max along axis 0 of [3;4] → [1;4] *)
  Schedule.reset ();
  let b = Tensor.from_float_list [3; 4] data in
  let m = Tensor.max_ ~axes:[0] b in
  check "partial max shape" (m.shape = [1; 4]);
  let mresult = Tensor.to_float_list m in
  check "partial max length" (List.length mresult = 4);
  check_float "partial max col0" (List.nth mresult 0) 9.0 1e-5;
  check_float "partial max col1" (List.nth mresult 1) 10.0 1e-5;
  check_float "partial max col2" (List.nth mresult 2) 11.0 1e-5;
  check_float "partial max col3" (List.nth mresult 3) 12.0 1e-5

(* ---- Test 19: Mean along axis ---- *)
let test_partial_mean () =
  Printf.printf "\n=== Partial-Axis Mean ===\n%!";
  Schedule.reset ();
  (* [2;4] tensor, mean along axis 1 → [2;1] *)
  let data = [2.0; 4.0; 6.0; 8.0;   (* row 0: mean=5 *)
              10.0; 20.0; 30.0; 40.0] (* row 1: mean=25 *) in
  let a = Tensor.from_float_list [2; 4] data in
  let m = Tensor.mean ~axes:[1] a in
  check "partial mean shape" (m.shape = [2; 1]);
  let result = Tensor.to_float_list m in
  check "partial mean length" (List.length result = 2);
  check_float "partial mean row0" (List.nth result 0) 5.0 1e-5;
  check_float "partial mean row1" (List.nth result 1) 25.0 1e-5

(* ---- Test 20: Multi-axis reduction ---- *)
let test_multi_axis_reduction () =
  Printf.printf "\n=== Multi-Axis Reduction ===\n%!";
  Schedule.reset ();
  (* [2;3;4] tensor, sum along axes [0;1] → [1;1;4]
     Data layout (row-major):
       slice 0: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
       slice 1: [[13,14,15,16], [17,18,19,20], [21,22,23,24]]
     col sums: [1+5+9+13+17+21, 2+6+10+14+18+22, 3+7+11+15+19+23, 4+8+12+16+20+24]
             = [66, 72, 78, 84] *)
  let data = List.init 24 (fun i -> Float.of_int (i + 1)) in
  let a = Tensor.from_float_list [2; 3; 4] data in
  let s = Tensor.sum ~axes:[0; 1] a in
  check "multi-axis sum shape" (s.shape = [1; 1; 4]);
  let result = Tensor.to_float_list s in
  check "multi-axis sum length" (List.length result = 4);
  check_float "multi-axis sum col0" (List.nth result 0) 66.0 1e-4;
  check_float "multi-axis sum col1" (List.nth result 1) 72.0 1e-4;
  check_float "multi-axis sum col2" (List.nth result 2) 78.0 1e-4;
  check_float "multi-axis sum col3" (List.nth result 3) 84.0 1e-4

(* ---- Test 21: Non-contiguous multi-axis reduction ---- *)
let test_noncontig_multi_axis () =
  Printf.printf "\n=== Non-Contiguous Multi-Axis Reduction ===\n%!";
  Schedule.reset ();
  (* [2;3;4] tensor, sum along axes [0;2] → [1;3;1]
     Data: 1..24 (row-major)
     slice 0: [[1,2,3,4], [5,6,7,8], [9,10,11,12]]
     slice 1: [[13,14,15,16], [17,18,19,20], [21,22,23,24]]
     For each j in 0..2:
       sum over i=0..1, k=0..3 of data[i][j][k]
     j=0: (1+2+3+4) + (13+14+15+16) = 10+58 = 68
     j=1: (5+6+7+8) + (17+18+19+20) = 26+74 = 100
     j=2: (9+10+11+12) + (21+22+23+24) = 42+90 = 132 *)
  let data = List.init 24 (fun i -> Float.of_int (i + 1)) in
  let a = Tensor.from_float_list [2; 3; 4] data in
  let s = Tensor.sum ~axes:[0; 2] a in
  check "noncontig shape" (s.shape = [1; 3; 1]);
  let result = Tensor.to_float_list s in
  check "noncontig length" (List.length result = 3);
  check_float "noncontig sum j=0" (List.nth result 0) 68.0 1e-4;
  check_float "noncontig sum j=1" (List.nth result 1) 100.0 1e-4;
  check_float "noncontig sum j=2" (List.nth result 2) 132.0 1e-4

(* ---- Test 22: Fused expression reduction ---- *)
let test_fused_reduction () =
  Printf.printf "\n=== Fused Expression Reduction ===\n%!";
  Schedule.reset ();
  (* sum(a + b) where a=[1,2,3,4] b=[10,20,30,40] *)
  let a = Tensor.from_float_list [4] [1.0; 2.0; 3.0; 4.0] in
  let b = Tensor.from_float_list [4] [10.0; 20.0; 30.0; 40.0] in
  let fused = Tensor.sqrt_ (Tensor.mul (Tensor.add a b) a) in
  let s = Tensor.sum fused in
  let result = Tensor.to_float_list s in
  (* sqrt((1+10)*1) + sqrt((2+20)*2) + sqrt((3+30)*3) + sqrt((4+40)*4) *)
  let expected = sqrt 11.0 +. sqrt 44.0 +. sqrt 99.0 +. sqrt 176.0 in
  check_float "fused sum(sqrt((a+b)*a))" (List.nth result 0) expected 1e-4

(* ---- Test 23: Chained reduction with different intermediate shapes ---- *)
let test_chained_reduction () =
  Printf.printf "\n=== Chained Reduction ===\n%!";
  Schedule.reset ();
  (* [2;3;4] tensor, sum(axis=2) → [2;3;1], then sum(axis=0) → [1;3;1]
     Data: 1..24

     Step 1: sum(axis=2) on [2;3;4]:
       row(0,0): 1+2+3+4 = 10
       row(0,1): 5+6+7+8 = 26
       row(0,2): 9+10+11+12 = 42
       row(1,0): 13+14+15+16 = 58
       row(1,1): 17+18+19+20 = 74
       row(1,2): 21+22+23+24 = 90
     → [2;3;1] = [10, 26, 42, 58, 74, 90]

     Step 2: sum(axis=0) on [2;3;1]:
       col0: 10+58 = 68
       col1: 26+74 = 100
       col2: 42+90 = 132
     → [1;3;1] = [68, 100, 132] *)
  let data = List.init 24 (fun i -> Float.of_int (i + 1)) in
  let a = Tensor.from_float_list [2; 3; 4] data in
  let inner = Tensor.sum ~axes:[2] a in    (* [2;3;1], numel=6 *)
  let outer = Tensor.sum ~axes:[0] inner in (* [1;3;1], numel=3 *)
  check "chained shape" (outer.shape = [1; 3; 1]);
  let result = Tensor.to_float_list outer in
  check "chained length" (List.length result = 3);
  check_float "chained sum col0" (List.nth result 0) 68.0 1e-4;
  check_float "chained sum col1" (List.nth result 1) 100.0 1e-4;
  check_float "chained sum col2" (List.nth result 2) 132.0 1e-4

(* ---- Test 24: Gradient computation ---- *)
let test_gradient () =
  Printf.printf "\n=== Gradient (backward) ===\n%!";
  Schedule.reset ();

  (* Test 1: d/dx sum(x * x) at x=[1,2,3] should give grad=[2,4,6] *)
  let x = Tensor.from_float_list [3] [1.0; 2.0; 3.0] in
  let x_sq = Tensor.mul x x in
  let loss = Tensor.sum x_sq in
  let grads = Tensor.backward loss [x] in
  check "grad count" (List.length grads = 1);
  let (_, dx) = List.hd grads in
  let dx_vals = Tensor.to_float_list dx in
  check "grad length" (List.length dx_vals = 3);
  check_float "d/dx sum(x*x)[0]" (List.nth dx_vals 0) 2.0 1e-5;
  check_float "d/dx sum(x*x)[1]" (List.nth dx_vals 1) 4.0 1e-5;
  check_float "d/dx sum(x*x)[2]" (List.nth dx_vals 2) 6.0 1e-5;

  (* Test 2: d/dx sum(x * y) should give grad_x=y, grad_y=x *)
  Schedule.reset ();
  let a = Tensor.from_float_list [4] [1.0; 2.0; 3.0; 4.0] in
  let b = Tensor.from_float_list [4] [10.0; 20.0; 30.0; 40.0] in
  let ab = Tensor.mul a b in
  let loss2 = Tensor.sum ab in
  let grads2 = Tensor.backward loss2 [a; b] in
  check "grad2 count" (List.length grads2 = 2);
  let (_, da) = List.nth grads2 0 in
  let (_, db) = List.nth grads2 1 in
  let da_vals = Tensor.to_float_list da in
  let db_vals = Tensor.to_float_list db in
  (* d/da sum(a*b) = b *)
  check_float "d/da sum(a*b)[0]" (List.nth da_vals 0) 10.0 1e-5;
  check_float "d/da sum(a*b)[1]" (List.nth da_vals 1) 20.0 1e-5;
  check_float "d/da sum(a*b)[2]" (List.nth da_vals 2) 30.0 1e-5;
  check_float "d/da sum(a*b)[3]" (List.nth da_vals 3) 40.0 1e-5;
  (* d/db sum(a*b) = a *)
  check_float "d/db sum(a*b)[0]" (List.nth db_vals 0) 1.0 1e-5;
  check_float "d/db sum(a*b)[1]" (List.nth db_vals 1) 2.0 1e-5;
  check_float "d/db sum(a*b)[2]" (List.nth db_vals 2) 3.0 1e-5;
  check_float "d/db sum(a*b)[3]" (List.nth db_vals 3) 4.0 1e-5

(* ---- Test 24b: Gradient through partial-axis reduction ---- *)
let test_gradient_partial_reduce () =
  Printf.printf "\n=== Gradient (partial reduce) ===\n%!";
  Schedule.reset ();
  (* x = [2;3] tensor, y = sum(x * x, axis=1) → [2;1]
     loss = sum(y) = sum of all x_i^2
     d/dx_i loss = 2 * x_i (same as full sum of squares)
     Data: [[1,2,3],[4,5,6]] *)
  let x = Tensor.from_float_list [2; 3] [1.0; 2.0; 3.0; 4.0; 5.0; 6.0] in
  let xsq = Tensor.mul x x in
  let y = Tensor.sum ~axes:[1] xsq in  (* [2;1] = [14, 77] *)
  let loss = Tensor.sum y in            (* scalar = 91 *)
  let grads = Tensor.backward loss [x] in
  check "partial reduce grad count" (List.length grads = 1);
  let (_, dx) = List.hd grads in
  let dx_vals = Tensor.to_float_list dx in
  check "partial reduce grad length" (List.length dx_vals = 6);
  (* d/dx_i = 2*x_i *)
  check_float "grad partial[0]" (List.nth dx_vals 0) 2.0 1e-5;
  check_float "grad partial[1]" (List.nth dx_vals 1) 4.0 1e-5;
  check_float "grad partial[2]" (List.nth dx_vals 2) 6.0 1e-5;
  check_float "grad partial[3]" (List.nth dx_vals 3) 8.0 1e-5;
  check_float "grad partial[4]" (List.nth dx_vals 4) 10.0 1e-5;
  check_float "grad partial[5]" (List.nth dx_vals 5) 12.0 1e-5

(* ---- Test 25: Simple gradient descent ---- *)
let test_gradient_descent () =
  Printf.printf "\n=== Gradient Descent ===\n%!";
  Schedule.reset ();
  (* Minimize f(x) = sum((x - target)^2) using gradient descent.
     target = [3.0; 5.0], starting from x = [0.0; 0.0].
     d/dx sum((x-t)^2) = 2*(x-t)
     With lr=0.1 and enough steps, x should converge toward target. *)
  let target = [3.0; 5.0] in
  let lr = 0.1 in
  let x_vals = ref [0.0; 0.0] in
  for _step = 0 to 29 do
    Schedule.reset ();
    let x = Tensor.from_float_list [2] !x_vals in
    let t = Tensor.from_float_list [2] target in
    let diff = Tensor.sub x t in
    let sq = Tensor.mul diff diff in
    let loss = Tensor.sum sq in
    let grads = Tensor.backward loss [x] in
    let (_, dx) = List.hd grads in
    let dx_vals = Tensor.to_float_list dx in
    (* SGD update: x = x - lr * grad *)
    x_vals := List.map2 (fun xv gv -> xv -. lr *. gv) !x_vals dx_vals
  done;
  (* After 30 steps of GD with lr=0.1 on quadratic: (1-0.2)^30 ≈ 0.001 *)
  check_float "gd x[0]→3.0" (List.nth !x_vals 0) 3.0 0.05;
  check_float "gd x[1]→5.0" (List.nth !x_vals 1) 5.0 0.05

(* ---- Test 25b: Max reduction backward ---- *)
let test_gradient_max_reduce () =
  Printf.printf "\n=== Gradient (max reduce) ===\n%!";
  Schedule.reset ();
  (* x = [1, 5, 3, 2], max(x) = 5 (at index 1, no ties)
     d/dx max(x) should be [0, 1, 0, 0] *)
  let x = Tensor.from_float_list [4] [1.0; 5.0; 3.0; 2.0] in
  let loss = Tensor.max_ x in
  let grads = Tensor.backward loss [x] in
  let (_, dx) = List.hd grads in
  let dx_vals = Tensor.to_float_list dx in
  check "max grad length" (List.length dx_vals = 4);
  check_float "max grad[0]" (List.nth dx_vals 0) 0.0 1e-5;
  check_float "max grad[1]" (List.nth dx_vals 1) 1.0 1e-5;
  check_float "max grad[2]" (List.nth dx_vals 2) 0.0 1e-5;
  check_float "max grad[3]" (List.nth dx_vals 3) 0.0 1e-5;

  (* Test with ties: x = [3, 7, 7, 2], max(x) = 7 (indices 1 and 2 tied)
     With tie-splitting, grad should be [0, 0.5, 0.5, 0] *)
  Schedule.reset ();
  let x2 = Tensor.from_float_list [4] [3.0; 7.0; 7.0; 2.0] in
  let loss2 = Tensor.max_ x2 in
  let grads2 = Tensor.backward loss2 [x2] in
  let (_, dx2) = List.hd grads2 in
  let dx2_vals = Tensor.to_float_list dx2 in
  check_float "max tie grad[0]" (List.nth dx2_vals 0) 0.0 1e-5;
  check_float "max tie grad[1]" (List.nth dx2_vals 1) 0.5 1e-5;
  check_float "max tie grad[2]" (List.nth dx2_vals 2) 0.5 1e-5;
  check_float "max tie grad[3]" (List.nth dx2_vals 3) 0.0 1e-5;

  (* Test partial-axis max: x = [[1,4,2],[5,3,6]], max(axis=1)
     → [[4],[6]]. Grad for [4] goes to position [0,1], grad for [6] goes to [1,2] *)
  Schedule.reset ();
  let x3 = Tensor.from_float_list [2; 3] [1.0; 4.0; 2.0; 5.0; 3.0; 6.0] in
  let y3 = Tensor.max_ ~axes:[1] x3 in  (* [2;1] = [4, 6] *)
  let loss3 = Tensor.sum y3 in
  let grads3 = Tensor.backward loss3 [x3] in
  let (_, dx3) = List.hd grads3 in
  let dx3_vals = Tensor.to_float_list dx3 in
  check "partial max grad length" (List.length dx3_vals = 6);
  check_float "partial max grad[0]" (List.nth dx3_vals 0) 0.0 1e-5;
  check_float "partial max grad[1]" (List.nth dx3_vals 1) 1.0 1e-5;
  check_float "partial max grad[2]" (List.nth dx3_vals 2) 0.0 1e-5;
  check_float "partial max grad[3]" (List.nth dx3_vals 3) 0.0 1e-5;
  check_float "partial max grad[4]" (List.nth dx3_vals 4) 0.0 1e-5;
  check_float "partial max grad[5]" (List.nth dx3_vals 5) 1.0 1e-5

(* ---- Test 25c: Gradient through sqrt and log2 ---- *)
let test_gradient_unary () =
  Printf.printf "\n=== Gradient (unary ops) ===\n%!";
  Schedule.reset ();
  (* d/dx sum(sqrt(x)) at x=[1,4,9] = [1/(2*1), 1/(2*2), 1/(2*3)] = [0.5, 0.25, 0.1667] *)
  let x = Tensor.from_float_list [3] [1.0; 4.0; 9.0] in
  let y = Tensor.sqrt_ x in
  let loss = Tensor.sum y in
  let grads = Tensor.backward loss [x] in
  let (_, dx) = List.hd grads in
  let dx_vals = Tensor.to_float_list dx in
  check_float "sqrt grad[0]" (List.nth dx_vals 0) 0.5 1e-5;
  check_float "sqrt grad[1]" (List.nth dx_vals 1) 0.25 1e-5;
  check_float "sqrt grad[2]" (List.nth dx_vals 2) (1.0 /. 6.0) 1e-4;

  (* d/dx sum(exp2(x)) at x=[0,1,2] = [ln2, 2*ln2, 4*ln2] *)
  Schedule.reset ();
  let x2 = Tensor.from_float_list [3] [0.0; 1.0; 2.0] in
  let y2 = Tensor.exp2_ x2 in
  let loss2 = Tensor.sum y2 in
  let grads2 = Tensor.backward loss2 [x2] in
  let (_, dx2) = List.hd grads2 in
  let dx2_vals = Tensor.to_float_list dx2 in
  let ln2 = Float.log 2.0 in
  check_float "exp2 grad[0]" (List.nth dx2_vals 0) (1.0 *. ln2) 1e-5;
  check_float "exp2 grad[1]" (List.nth dx2_vals 1) (2.0 *. ln2) 1e-5;
  check_float "exp2 grad[2]" (List.nth dx2_vals 2) (4.0 *. ln2) 1e-4

(* ---- Test 25d: Gradient through chained reductions with broadcast ---- *)
let test_gradient_chained_reduce () =
  Printf.printf "\n=== Gradient (chained reduce broadcast) ===\n%!";
  Schedule.reset ();
  (* x = [2;3] tensor, y = sum(x, axis=1) → [2;1],
     z = y * y → [2;1] (elementwise square of partial sums),
     loss = sum(z).
     Forward: x = [[1,2,3],[4,5,6]], y = [6, 15], z = [36, 225], loss = 261.
     d/dz sum(z) = [1, 1]
     d/dy y*y = 2y → [12, 30]
     d/dx sum(x, axis=1) = expand [12, 30] to [[12,12,12],[30,30,30]]
     This exercises broadcast indexing: the y buffer (size 2) must be correctly
     broadcast-indexed when used in a size-6 kernel. *)
  let x = Tensor.from_float_list [2; 3] [1.0; 2.0; 3.0; 4.0; 5.0; 6.0] in
  let y = Tensor.sum ~axes:[1] x in          (* [2;1] = [6, 15] *)
  let z = Tensor.mul y y in                   (* [2;1] = [36, 225] *)
  let loss = Tensor.sum z in                  (* scalar = 261 *)
  let grads = Tensor.backward loss [x] in
  let (_, dx) = List.hd grads in
  let dx_vals = Tensor.to_float_list dx in
  check "chained reduce broadcast grad length" (List.length dx_vals = 6);
  (* d/dx = 2 * sum(x, axis=1) broadcast back to [2;3] *)
  check_float "chained reduce bcast[0]" (List.nth dx_vals 0) 12.0 1e-4;
  check_float "chained reduce bcast[1]" (List.nth dx_vals 1) 12.0 1e-4;
  check_float "chained reduce bcast[2]" (List.nth dx_vals 2) 12.0 1e-4;
  check_float "chained reduce bcast[3]" (List.nth dx_vals 3) 30.0 1e-4;
  check_float "chained reduce bcast[4]" (List.nth dx_vals 4) 30.0 1e-4;
  check_float "chained reduce bcast[5]" (List.nth dx_vals 5) 30.0 1e-4

(* ---- Test 25e: Gradient through sin ---- *)
let test_gradient_sin () =
  Printf.printf "\n=== Gradient (sin) ===\n%!";
  Schedule.reset ();
  (* d/dx sum(sin(x)) at x=[0, pi/4, pi/2] = cos(x) = [1, sqrt(2)/2, 0] *)
  let pi = Float.pi in
  let x = Tensor.from_float_list [3] [0.0; pi /. 4.0; pi /. 2.0] in
  let y = Tensor.sin_ x in
  let loss = Tensor.sum y in
  let grads = Tensor.backward loss [x] in
  let (_, dx) = List.hd grads in
  let dx_vals = Tensor.to_float_list dx in
  check_float "sin grad[0]" (List.nth dx_vals 0) 1.0 1e-4;
  check_float "sin grad[1]" (List.nth dx_vals 1) (Float.sqrt 2.0 /. 2.0) 1e-4;
  check_float "sin grad[2]" (List.nth dx_vals 2) 0.0 1e-4

(* ---- Test 26: Input buffer broadcast via step-1 shape metadata ---- *)
let test_input_broadcast () =
  Printf.printf "\n=== Input Buffer Broadcast ===\n%!";
  Schedule.reset ();
  (* Create a [1;3] tensor and a [2;3] tensor, expand [1;3] to [2;3], add them.
     This exercises the step-1 copyin path storing shape [1;3] so the downstream
     kernel can use broadcast_index instead of the ratio heuristic. *)
  let a = Tensor.from_float_list [1; 3] [10.0; 20.0; 30.0] in
  let b = Tensor.from_float_list [2; 3] [1.0; 2.0; 3.0; 4.0; 5.0; 6.0] in
  let a_exp = Tensor.expand a [2; 3] in
  let result = Tensor.add a_exp b in
  let vals = Tensor.to_float_list result in
  check "input bcast length" (List.length vals = 6);
  (* [10+1, 20+2, 30+3, 10+4, 20+5, 30+6] = [11, 22, 33, 14, 25, 36] *)
  check_float "input bcast[0]" (List.nth vals 0) 11.0 1e-6;
  check_float "input bcast[1]" (List.nth vals 1) 22.0 1e-6;
  check_float "input bcast[2]" (List.nth vals 2) 33.0 1e-6;
  check_float "input bcast[3]" (List.nth vals 3) 14.0 1e-6;
  check_float "input bcast[4]" (List.nth vals 4) 25.0 1e-6;
  check_float "input bcast[5]" (List.nth vals 5) 36.0 1e-6

(* ---- Test 27: Leading-axis broadcast from input buffer ---- *)
let test_leading_axis_broadcast () =
  Printf.printf "\n=== Leading-Axis Broadcast ===\n%!";
  Schedule.reset ();
  (* Create [3;1] and [3;2] tensors. Expand [3;1] to [3;2] and multiply.
     This tests the case codex flagged: leading-axis broadcast where the
     ratio heuristic gives wrong results but stride-based indexing is correct. *)
  let a = Tensor.from_float_list [3; 1] [2.0; 3.0; 4.0] in
  let b = Tensor.from_float_list [3; 2] [1.0; 10.0; 1.0; 10.0; 1.0; 10.0] in
  let a_exp = Tensor.expand a [3; 2] in
  let result = Tensor.mul a_exp b in
  let vals = Tensor.to_float_list result in
  check "leading bcast length" (List.length vals = 6);
  (* [2*1, 2*10, 3*1, 3*10, 4*1, 4*10] = [2, 20, 3, 30, 4, 40] *)
  check_float "leading bcast[0]" (List.nth vals 0) 2.0 1e-6;
  check_float "leading bcast[1]" (List.nth vals 1) 20.0 1e-6;
  check_float "leading bcast[2]" (List.nth vals 2) 3.0 1e-6;
  check_float "leading bcast[3]" (List.nth vals 3) 30.0 1e-6;
  check_float "leading bcast[4]" (List.nth vals 4) 4.0 1e-6;
  check_float "leading bcast[5]" (List.nth vals 5) 40.0 1e-6

(* ---- Test 28: Log2 gradient ---- *)
let test_gradient_log2 () =
  Printf.printf "\n=== Gradient (log2) ===\n%!";
  Schedule.reset ();
  (* d/dx sum(log2(x)) = 1/(x * ln(2)) *)
  let x = Tensor.from_float_list [3] [1.0; 2.0; 4.0] in
  let y = Tensor.log2_ x in
  let loss = Tensor.sum y in
  let grads = Tensor.backward loss [x] in
  let (_, dx) = List.hd grads in
  let dx_vals = Tensor.to_float_list dx in
  let ln2 = Float.log 2.0 in
  check_float "log2 grad[0]" (List.nth dx_vals 0) (1.0 /. (1.0 *. ln2)) 1e-4;
  check_float "log2 grad[1]" (List.nth dx_vals 1) (1.0 /. (2.0 *. ln2)) 1e-4;
  check_float "log2 grad[2]" (List.nth dx_vals 2) (1.0 /. (4.0 *. ln2)) 1e-4

(* ---- Test 29: Permute gradient ---- *)
let test_gradient_permute () =
  Printf.printf "\n=== Gradient (permute) ===\n%!";
  Schedule.reset ();
  (* x is [2;3], permute to [3;2], sum all → scalar.
     d/dx sum(permute(x, [1;0])) = ones([2;3]) since sum gradient is all-1s
     and permute gradient is inverse permute (which is [1;0] again for swap). *)
  let x = Tensor.from_float_list [2; 3] [1.0; 2.0; 3.0; 4.0; 5.0; 6.0] in
  let xp = Tensor.permute x [1; 0] in
  let loss = Tensor.sum xp in
  let grads = Tensor.backward loss [x] in
  let (_, dx) = List.hd grads in
  let dx_vals = Tensor.to_float_list dx in
  check "permute grad length" (List.length dx_vals = 6);
  List.iteri (fun i v ->
    check_float (Printf.sprintf "permute grad[%d]" i) v 1.0 1e-6
  ) dx_vals

(* ---- Test 30: Where + comparison ops (elementwise max via where) ---- *)
let test_where_cmp () =
  Printf.printf "\n=== Where + Comparison ===\n%!";
  Schedule.reset ();
  (* where(a < b, b, a) = elementwise max(a, b) *)
  let a = Tensor.from_float_list [4] [3.0; 1.0; 4.0; 2.0] in
  let b = Tensor.from_float_list [4] [2.0; 5.0; 1.0; 2.0] in
  let cond = Tensor.lt a b in  (* [true, false, false, false] = [1, 0, 0, 0] *)
  let result = Tensor.where_ cond b a in  (* select b where a<b, else a *)
  let vals = Tensor.to_float_list result in
  check "where length" (List.length vals = 4);
  (* max(3,2)=3, max(1,5)=5, max(4,1)=4, max(2,2)=2 *)
  check_float "where[0]" (List.nth vals 0) 3.0 1e-6;
  check_float "where[1]" (List.nth vals 1) 5.0 1e-6;
  check_float "where[2]" (List.nth vals 2) 4.0 1e-6;
  check_float "where[3]" (List.nth vals 3) 2.0 1e-6

(* ---- Test 31: Cast bool→float forward ---- *)
let test_cast_forward () =
  Printf.printf "\n=== Cast Forward ===\n%!";
  Schedule.reset ();
  (* cast(a < b, float32) should give [1.0, 0.0, 0.0, 1.0] *)
  let a = Tensor.from_float_list [4] [1.0; 5.0; 3.0; 0.0] in
  let b = Tensor.from_float_list [4] [2.0; 3.0; 3.0; 1.0] in
  let cond = Tensor.lt a b in
  let cond_f = Tensor.cast Dtype.float32 cond in
  let vals = Tensor.to_float_list cond_f in
  check "cast length" (List.length vals = 4);
  check_float "cast[0]" (List.nth vals 0) 1.0 1e-6;
  check_float "cast[1]" (List.nth vals 1) 0.0 1e-6;
  check_float "cast[2]" (List.nth vals 2) 0.0 1e-6;
  check_float "cast[3]" (List.nth vals 3) 1.0 1e-6

(* ---- Test 32: Expand gradient ---- *)
let test_gradient_expand () =
  Printf.printf "\n=== Gradient (expand) ===\n%!";
  Schedule.reset ();
  (* x is [1;3], expand to [2;3], then sum all.
     d/dx sum(expand(x, [2;3])) = [2, 2, 2] because each element of x
     appears twice in the expansion, so gradient is 2 per element. *)
  let x = Tensor.from_float_list [1; 3] [10.0; 20.0; 30.0] in
  let x_exp = Tensor.expand x [2; 3] in
  let loss = Tensor.sum x_exp in
  let grads = Tensor.backward loss [x] in
  let (_, dx) = List.hd grads in
  let dx_vals = Tensor.to_float_list dx in
  check "expand grad length" (List.length dx_vals = 3);
  check_float "expand grad[0]" (List.nth dx_vals 0) 2.0 1e-6;
  check_float "expand grad[1]" (List.nth dx_vals 1) 2.0 1e-6;
  check_float "expand grad[2]" (List.nth dx_vals 2) 2.0 1e-6

(* ---- Test 33: Where gradient ---- *)
let test_gradient_where () =
  Printf.printf "\n=== Gradient (where) ===\n%!";
  Schedule.reset ();
  (* relu(x) = where(x > 0, x, 0). d/dx sum(relu(x)) = where(x > 0, 1, 0) *)
  let x = Tensor.from_float_list [4] [~-.2.0; 3.0; ~-.1.0; 5.0] in
  let zero = Tensor.zeros [4] in
  let cond = Tensor.lt zero x in  (* 0 < x = [false, true, false, true] *)
  let relu = Tensor.where_ cond x zero in
  let loss = Tensor.sum relu in
  let grads = Tensor.backward loss [x] in
  let (_, dx) = List.hd grads in
  let dx_vals = Tensor.to_float_list dx in
  check "where grad length" (List.length dx_vals = 4);
  check_float "relu grad[0]" (List.nth dx_vals 0) 0.0 1e-6;
  check_float "relu grad[1]" (List.nth dx_vals 1) 1.0 1e-6;
  check_float "relu grad[2]" (List.nth dx_vals 2) 0.0 1e-6;
  check_float "relu grad[3]" (List.nth dx_vals 3) 1.0 1e-6

(* ---- Test 34: Expand + mul forward (non-trivial broadcast expression) ---- *)
let test_expand_mul () =
  Printf.printf "\n=== Expand + Mul Forward ===\n%!";
  Schedule.reset ();
  (* x=[1;3], expand to [2;3], multiply with y=[2;3].
     Tests that expand + elementwise works correctly through the scheduler. *)
  let x = Tensor.from_float_list [1; 3] [2.0; 3.0; 4.0] in
  let y = Tensor.from_float_list [2; 3] [1.0; 1.0; 1.0; 10.0; 10.0; 10.0] in
  let x_exp = Tensor.expand x [2; 3] in
  let result = Tensor.mul x_exp y in
  let vals = Tensor.to_float_list result in
  check "expand_mul length" (List.length vals = 6);
  check_float "expand_mul[0]" (List.nth vals 0) 2.0 1e-6;
  check_float "expand_mul[1]" (List.nth vals 1) 3.0 1e-6;
  check_float "expand_mul[2]" (List.nth vals 2) 4.0 1e-6;
  check_float "expand_mul[3]" (List.nth vals 3) 20.0 1e-6;
  check_float "expand_mul[4]" (List.nth vals 4) 30.0 1e-6;
  check_float "expand_mul[5]" (List.nth vals 5) 40.0 1e-6

(* ---- Test 35: Matmul forward ---- *)
let test_matmul_forward () =
  Printf.printf "\n=== Matmul Forward ===\n%!";
  Schedule.reset ();
  (* a = [[1, 2],    b = [[5, 6],     a @ b = [[1*5+2*7, 1*6+2*8],
          [3, 4]]         [7, 8]]               [3*5+4*7, 3*6+4*8]]
                                             = [[19, 22],
                                                [43, 50]] *)
  let a = Tensor.from_float_list [2; 2] [1.0; 2.0; 3.0; 4.0] in
  let b = Tensor.from_float_list [2; 2] [5.0; 6.0; 7.0; 8.0] in
  let c = Tensor.matmul a b in
  let vals = Tensor.to_float_list c in
  check "matmul shape" (c.shape = [2; 2]);
  check "matmul length" (List.length vals = 4);
  check_float "matmul[0,0]" (List.nth vals 0) 19.0 1e-4;
  check_float "matmul[0,1]" (List.nth vals 1) 22.0 1e-4;
  check_float "matmul[1,0]" (List.nth vals 2) 43.0 1e-4;
  check_float "matmul[1,1]" (List.nth vals 3) 50.0 1e-4

(* ---- Test 36: Matmul non-square ---- *)
let test_matmul_nonsquare () =
  Printf.printf "\n=== Matmul Non-Square ===\n%!";
  Schedule.reset ();
  (* a = [[1, 2, 3]]  (1x3)    b = [[1],    a @ b = [[1+4+9]] = [[14]]
                                     [2],
                                     [3]]    (3x1) *)
  let a = Tensor.from_float_list [1; 3] [1.0; 2.0; 3.0] in
  let b = Tensor.from_float_list [3; 1] [1.0; 2.0; 3.0] in
  let c = Tensor.matmul a b in
  check "matmul_ns shape" (c.shape = [1; 1]);
  let vals = Tensor.to_float_list c in
  check_float "matmul_ns dot" (List.nth vals 0) 14.0 1e-4;
  (* Also test 2x3 @ 3x2 *)
  Schedule.reset ();
  let a2 = Tensor.from_float_list [2; 3] [1.0; 2.0; 3.0; 4.0; 5.0; 6.0] in
  let b2 = Tensor.from_float_list [3; 2] [1.0; 4.0; 2.0; 5.0; 3.0; 6.0] in
  let c2 = Tensor.matmul a2 b2 in
  check "matmul_ns2 shape" (c2.shape = [2; 2]);
  let vals2 = Tensor.to_float_list c2 in
  (* [1*1+2*2+3*3, 1*4+2*5+3*6] = [14, 32]
     [4*1+5*2+6*3, 4*4+5*5+6*6] = [32, 77] *)
  check_float "matmul_ns2[0,0]" (List.nth vals2 0) 14.0 1e-4;
  check_float "matmul_ns2[0,1]" (List.nth vals2 1) 32.0 1e-4;
  check_float "matmul_ns2[1,0]" (List.nth vals2 2) 32.0 1e-4;
  check_float "matmul_ns2[1,1]" (List.nth vals2 3) 77.0 1e-4

(* ---- Test 37: Matmul gradient ---- *)
let test_matmul_gradient () =
  Printf.printf "\n=== Matmul Gradient ===\n%!";
  Schedule.reset ();
  (* loss = sum(a @ b). For a[N,K] @ b[K,M]:
     d(loss)/d(a) = ones[N,M] @ b^T = ones @ b^T
     d(loss)/d(b) = a^T @ ones[N,M] = a^T @ ones
     With a=[[1,2],[3,4]] b=[[5,6],[7,8]]:
     d/da = ones[2,2] @ [[5,7],[6,8]] = [[11,15],[11,15]]
     d/db = [[1,3],[2,4]] @ ones[2,2] = [[4,4],[6,6]] *)
  let a = Tensor.from_float_list [2; 2] [1.0; 2.0; 3.0; 4.0] in
  let b = Tensor.from_float_list [2; 2] [5.0; 6.0; 7.0; 8.0] in
  let c = Tensor.matmul a b in
  let loss = Tensor.sum c in
  let grads = Tensor.backward loss [a; b] in
  let (_, da) = List.nth grads 0 in
  let (_, db) = List.nth grads 1 in
  let da_vals = Tensor.to_float_list da in
  let db_vals = Tensor.to_float_list db in
  check "matmul grad a length" (List.length da_vals = 4);
  check "matmul grad b length" (List.length db_vals = 4);
  (* d/da = [[5+6, 7+8], [5+6, 7+8]] = [[11, 15], [11, 15]] *)
  check_float "da[0,0]" (List.nth da_vals 0) 11.0 1e-4;
  check_float "da[0,1]" (List.nth da_vals 1) 15.0 1e-4;
  check_float "da[1,0]" (List.nth da_vals 2) 11.0 1e-4;
  check_float "da[1,1]" (List.nth da_vals 3) 15.0 1e-4;
  (* d/db = [[1+3, 1+3], [2+4, 2+4]] = [[4, 4], [6, 6]] *)
  check_float "db[0,0]" (List.nth db_vals 0) 4.0 1e-4;
  check_float "db[0,1]" (List.nth db_vals 1) 4.0 1e-4;
  check_float "db[1,0]" (List.nth db_vals 2) 6.0 1e-4;
  check_float "db[1,1]" (List.nth db_vals 3) 6.0 1e-4

(* ---- Test 38: Linear layer (matmul + gradient descent) ---- *)
let test_linear_layer () =
  Printf.printf "\n=== Linear Layer Training ===\n%!";
  (* Train a single linear layer: y = x @ w to fit target.
     x = [[1, 0], [0, 1], [1, 1]]  (3 samples, 2 features)
     target = [[2], [3], [5]]  (3 samples, 1 output)
     This should learn w ≈ [[2], [3]] since target = x @ [2, 3]^T.
     We use MSE loss and SGD.
     Note: x, target, and w are recreated each step because
     Schedule.reset() clears the buffer_data table. *)
  let x_data = [1.0; 0.0; 0.0; 1.0; 1.0; 1.0] in
  let target_data = [2.0; 3.0; 5.0] in
  let lr = 0.05 in
  let w_vals = ref [0.1; 0.1] in
  (* Train for 100 steps *)
  for _step = 1 to 100 do
    Schedule.reset ();
    let x = Tensor.from_float_list [3; 2] x_data in
    let target = Tensor.from_float_list [3; 1] target_data in
    let w = Tensor.from_float_list [2; 1] !w_vals in
    let pred = Tensor.matmul x w in  (* [3;1] *)
    let diff = Tensor.sub pred target in  (* [3;1] *)
    let loss = Tensor.mean (Tensor.mul diff diff) in  (* scalar *)
    let grads = Tensor.backward loss [w] in
    let (_, dw) = List.hd grads in
    let dw_v = Tensor.to_float_list dw in
    (* SGD update *)
    w_vals := List.map2 (fun wi dwi -> wi -. lr *. dwi) !w_vals dw_v
  done;
  Printf.printf "    w_final = [%s]\n%!"
    (String.concat ", " (List.map (Printf.sprintf "%.4f") !w_vals));
  (* Should converge to w ≈ [2, 3] *)
  check_float "w[0] ≈ 2" (List.nth !w_vals 0) 2.0 0.1;
  check_float "w[1] ≈ 3" (List.nth !w_vals 1) 3.0 0.1

(* ---- Test 39: CUDA backend registration ---- *)
let test_cuda_backend () =
  Printf.printf "\n=== CUDA Backend ===\n%!";
  (* Verify CUDA backend is recognized and returns a module *)
  let module B = (val Device.get_backend "CUDA" : Device.Backend) in
  check "cuda backend name" (B.device_name = "CUDA");
  (* Verify CUDA rendering produces valid kernel source *)
  let in_param = Uop.param 0 (Dtype.ptr Dtype.float32) in
  let out_param = Uop.param 1 (Dtype.ptr Dtype.float32) in
  let i = Uop.range (Uop.const_int Dtype.int32 4) [0; 0] in
  let v = Uop.load (Uop.index in_param i) in
  let r = Uop.mul v (Uop.const Dtype.float32 2.0) in
  let st = Uop.store (Uop.index out_param i) r in
  let end_r = Uop.end_ i in
  let kernel = Uop.sink ~name:"cuda_test" [st; end_r] in
  let uops = Uop.toposort1 kernel in
  let pspec = Cstyle.render_uops (Cstyle.cuda_config ~arch:"sm_80") uops in
  check "cuda src has __global__" (String.length pspec.src > 0
    && (try ignore (Str.search_forward (Str.regexp_string "__global__") pspec.src 0); true with Not_found -> false));
  check "cuda src has extern C" (try ignore (Str.search_forward (Str.regexp_string {|extern "C"|}) pspec.src 0); true with Not_found -> false)

let run_test name f =
  try f () with e ->
    incr fail_count;
    Printf.printf "  ERROR in %s: %s\n%!" name (Printexc.to_string e)

(* ---- Main ---- *)
let () =
  Printf.printf "tinygrad_ml end-to-end tests\n%!";
  Printf.printf "============================\n%!";
  (* Low-level UOp kernel tests *)
  run_test "vector_add" test_vector_add;
  run_test "fused_ops" test_fused_ops;
  run_test "reduction" test_reduction;
  run_test "unary_ops" test_unary_ops;
  run_test "multi_backend_render" test_multi_backend_render;
  run_test "metal_execution" test_metal_execution;
  (* Tensor API tests *)
  run_test "tensor_roundtrip" test_tensor_roundtrip;
  run_test "tensor_compute" test_tensor_compute;
  run_test "tensor_compute_metal" test_tensor_compute_metal;
  run_test "tensor_unary_ops" test_tensor_unary_ops;
  run_test "tensor_div" test_tensor_div;
  run_test "tensor_deep_chain" test_tensor_deep_chain;
  run_test "metal_chain" test_metal_chain;
  run_test "tensor_const" test_tensor_const;
  run_test "tensor_sum" test_tensor_sum;
  run_test "tensor_max" test_tensor_max;
  run_test "tensor_mean" test_tensor_mean;
  run_test "partial_reduction" test_partial_reduction;
  run_test "partial_mean" test_partial_mean;
  run_test "multi_axis_reduction" test_multi_axis_reduction;
  run_test "noncontig_multi_axis" test_noncontig_multi_axis;
  run_test "fused_reduction" test_fused_reduction;
  run_test "chained_reduction" test_chained_reduction;
  run_test "gradient" test_gradient;
  run_test "gradient_partial_reduce" test_gradient_partial_reduce;
  run_test "gradient_descent" test_gradient_descent;
  run_test "gradient_max_reduce" test_gradient_max_reduce;
  run_test "gradient_unary" test_gradient_unary;
  run_test "gradient_chained_reduce" test_gradient_chained_reduce;
  run_test "gradient_sin" test_gradient_sin;
  run_test "input_broadcast" test_input_broadcast;
  run_test "leading_axis_broadcast" test_leading_axis_broadcast;
  run_test "gradient_log2" test_gradient_log2;
  run_test "gradient_permute" test_gradient_permute;
  run_test "where_cmp" test_where_cmp;
  run_test "cast_forward" test_cast_forward;
  run_test "gradient_expand" test_gradient_expand;
  run_test "gradient_where" test_gradient_where;
  run_test "expand_mul" test_expand_mul;
  run_test "matmul_forward" test_matmul_forward;
  run_test "matmul_nonsquare" test_matmul_nonsquare;
  run_test "matmul_gradient" test_matmul_gradient;
  run_test "linear_layer" test_linear_layer;
  run_test "cuda_backend" test_cuda_backend;
  Printf.printf "\n============================\n%!";
  Printf.printf "Results: %d passed, %d failed\n%!" !pass_count !fail_count;
  if !fail_count > 0 then exit 1

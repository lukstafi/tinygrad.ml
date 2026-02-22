Port tinygrad: ~/tinygrad/ to OCaml. Where reasonable, minimize how much of the frontend niceties are ported if functionality is not significantly reduced. Port a CPU C backend and the CUDA and Metal MSL backends using the `cudajit` ~/ocaml-cudajit/ and `metal` ~/ocaml-metal/ packages, do not port the other lower-level backends. Balance faithfulness to the tinygrad source code with educational value and using idiomatic OCaml rather than trying to emulate Python semantics.

## Claude round 2 decisions

- Added Metal GPU backend using `metal` OCaml package with shared-storage buffers, kernel caching, MSL compilation via Metal.Library, compute pipeline dispatch.
- Fixed Tensor data pipeline: `from_float_list` stores data via `Schedule.store_buffer_data`, `to_float_list` retrieves from realized Device.buffer via `Device.copyout_floats`.
- Implemented minimal Schedule: walks UOp graph to find unrealized BUFFER nodes, creates Copyin exec items that allocate device buffers and copy data.
- Implemented Realize: executes Copyin items (copies float data into device buffers) and Kernel items (renders, compiles, executes).
- CUDA backend not implemented (cudajit package not installed on this machine). Renderer generates valid CUDA code but no runtime execution.
- Metal backend uses loop-based kernels (1 threadgroup) since the scheduler doesn't yet emit SPECIAL ops for GPU thread indexing. Correct but not optimal.
- Extended CPU exec dispatch to handle 4-5 buffer arguments.

## Claude round 3 decisions

- Implemented full schedule/realize pipeline for lazy tensor compute: `lower_to_kernel` converts ALU expression graphs into executable kernel UOps with PARAM→INDEX→LOAD chains.
- Fixed `to_float_list` to fail hard on missing realized buffer instead of returning silent zeros.
- Fixed test harness: exceptions now increment `fail_count` via `run_test` wrapper.
- Fixed chained compute: `tensor.uop` is mutable, `realize` replaces it with a BUFFER node pointing to computed results.

## Claude round 4 decisions

- Fixed kernel name collision bug: CPU backend cached compiled `.so` by path, which was the same for all kernels named "tensor_kernel". Fixed by giving each kernel a unique name (`tk_0`, `tk_1`, ...) and unique `.so` filenames.
- Fixed `numel` inference: now passed from tensor shape via `create_schedule ~numel` instead of heuristic max over buffer sizes.
- Added `Schedule.reset()` for test isolation (clears `buffer_data` and `realized_buffers` tables).
- Refactored `rebuild_expr` into standalone function for clarity.
- Added CONST-only graph support: `full/zeros/ones` tensors now realize through a fill kernel (no input buffers, just CONST→STORE).
- Added CAST handling in the rebuild function.
- Expanded e2e tests from 47 to 73: neg, reciprocal, exp2, div (mul+reciprocal fusion), 3-stage chained compute, Metal chained compute, const/zeros/ones tensors.
- CUDA backend still not wired (cudajit not installed on macOS, no NVIDIA GPU).

## Claude round 5 decisions

- Implemented Tensor reduction kernels through the scheduler: `sum`, `max`, `mean` now compile and execute through the full UOp→render→compile→exec pipeline.
- Reduction lowering generates accumulator-based kernels: init output to identity (0 for ADD, -inf for MAX), loop over input, accumulate.
- Multi-pass scheduling: `create_schedule` first schedules inner REDUCE_AXIS nodes (for `mean = sum * 1/n`), then the outer elementwise expression. Realized REDUCE_AXIS nodes are treated as input buffers in subsequent kernels.
- Made `test_e2e` a proper `(test)` in dune so `dune test` catches regressions.
- Completed `Schedule.reset` to also reset `kernel_counter` for deterministic test isolation.
- Total tests: 96 unit + 80 e2e = 176 all passing.

## Claude round 6 decisions

- Added CUDA backend stub to `device.ml`: `module CUDA : Backend` with clear error messages ("cudajit not installed") for all operations except `synchronize`. Wired into `get_backend` dispatch.
- Fixed `input_numel` inference for reductions: `tensor.ml` now scans the UOp graph for `Dtype.Ptr` buffer sizes to determine the pre-reduction element count, passing it to `create_schedule ~input_numel`. This replaces heuristic assumptions when the tensor shape alone can't reveal the source size.
- Fixed `Dtype.Ptr` pattern match bug: constructor field order is `Ptr(base, addr_space, size)`, not `Ptr(base, size, addr_space)`. The round 5 code had the wrong field order, causing a type error.
- Added CUDA backend e2e test: verifies stub module name and that rendered CUDA source has `__global__` and `extern "C"` markers.
- Total tests: 96 unit + 83 e2e = 179 all passing.

## Claude round 7 decisions

- **Per-REDUCE_AXIS input_numel**: Moved reduction sizing from `tensor.ml` into `schedule.ml`. Each REDUCE_AXIS node now infers its own `input_numel` from source buffer sizes via `infer_reduce_input_numel`. Removed the fragile root-level `~input_numel` parameter from `create_schedule`.
- **Partial-axis reductions**: Extended `lower_reduce_kernel` to handle `output_numel > 1`. Uses stride-aware indexing: `flat_idx = (o / inner_numel) * stride + r * inner_numel + (o % inner_numel)` where `inner_numel` = product of dims after the reduced axis. This correctly handles both last-axis reductions (`sum(axis=-1)`) and non-last-axis reductions (`max(axis=0)`).
- **Accumulator linearization fix**: Inner-loop accumulator LOAD depends on the inner loop variable via `o + (r - r)` trick, forcing the renderer to place it inside the inner loop (same pattern used in full reductions).
- **CPU allocator GC fix**: Added `ba_roots` hashtable to keep Bigarray backing memory alive while raw pointers reference it. Previously, the GC could collect the Bigarray and reassign the same memory to a new allocation, causing buffer aliasing. This was a latent bug affecting all multi-kernel pipelines.
- **Scheduler takes `~output_shape`** instead of just `~numel`, enabling stride computation for partial reductions from the tensor's shape + reduce axes.
- **TINYGRAD_DEBUG env var**: `realize.ml` prints generated kernel source when `TINYGRAD_DEBUG=1`.
- Total tests: 96 unit + 98 e2e = 194 all passing.

## Claude round 8 decisions

- **Explicit src_shape in Axis_arg**: Extended `Axis_arg` from `(axes, reduce_op)` to `(axes, reduce_op, src_shape)`. Tensor.reduce now passes `~src_shape:t.shape` into the UOp, and the scheduler extracts the explicit source shape directly from the `Axis_arg` instead of heuristically reconstructing it from the output shape and reduce axes. This makes multi-axis reductions correct by construction.
- **Multi-axis reduction support**: With explicit `src_shape`, multi-axis reductions (e.g., `sum(axes=[0;1])` on `[2;3;4]`) now work correctly. The `inner_numel` calculation in `lower_reduce_kernel` uses the max reduced axis from `src_shape` to determine stride layout.
- **Updated gradient.ml**: Pattern match on `Axis_arg` updated to 3-tuple.
- **Fallback preserved**: If `src_shape` is empty (legacy UOps), the scheduler falls back to heuristic inference from buffer sizes, maintaining backward compatibility.
- Total tests: 96 unit + 104 e2e = 200 all passing.

## Claude round 9 decisions

- **General non-contiguous multi-axis reduction indexing**: Replaced the single-axis-optimized `inner_numel` stride computation with a general approach that decomposes output index `o` and reduce index `r` into per-axis indices using `idiv`/`mod`, then combines them with input strides to compute the flat input index. Handles arbitrary axis combinations (e.g., `[0;2]` on `[2;3;4]`).
- **Fast path preserved**: Single-axis and contiguous-trailing-axes reductions still use the efficient `inner_numel` shortcut. The general decomposition path only activates for truly non-contiguous axis sets.
- **Added non-contiguous multi-axis reduction test**: `sum(axes=[0;2])` on `[2;3;4]` → `[1;3;1]` with expected values `[68, 100, 132]`.
- **Added fused expression reduction test**: `sum(sqrt((a+b)*a))` verifies expression evaluation inside reduction loop.
- Total tests: 96 unit + 110 e2e = 206 all passing.

## Claude round 10 decisions

- **Per-node output_numel for chained reductions**: Each `REDUCE_AXIS` node now computes its own `output_numel` from its `src_shape` and `reduce_axes`, instead of reusing the root tensor's numel. This fixes chained reductions like `sum(sum(x, axis=2), axis=0)` where intermediate output shapes differ from the final shape.
- **Realized REDUCE_AXIS as reduction inputs**: `lower_reduce_kernel` now scans for realized `REDUCE_AXIS` nodes (not just `BUFFER` nodes) as input sources, allowing reductions to chain through realized intermediate results.
- **Added chained reduction test**: `sum(axis=2)` then `sum(axis=0)` on `[2;3;4]` → `[1;3;1]` verifying that intermediate `[2;3;1]` is correctly used as input to the second reduction.
- Total tests: 96 unit + 115 e2e = 211 all passing.

## Claude round 11 decisions

- **Tensor-level `backward` API**: Added `Tensor.backward loss targets` that performs reverse-mode autodiff through the UOp graph and returns gradient tensors. Takes a scalar loss tensor and list of target tensors, returns `(target, gradient_tensor)` pairs that can be realized and used for parameter updates.
- **Fixed RESHAPE gradient**: Now correctly reshapes the incoming gradient back to the source shape using shape metadata from the UOp arg.
- **Fixed EXPAND gradient**: Now reduces (sums) over expanded dimensions and reshapes back to the source shape, correctly implementing the transpose of broadcast.
- **Gradient descent test**: End-to-end test optimizing `f(x) = sum((x - target)^2)` using 30 steps of SGD. Demonstrates the full forward → backward → update pipeline converging from `[0,0]` to `[3,5]` within tolerance 0.05.
- **Gradient computation tests**: Verified `d/dx sum(x*x) = 2x` and `d/da sum(a*b) = b, d/db sum(a*b) = a`.
- Total tests: 96 unit + 131 e2e = 227 all passing.

## Claude round 12 decisions

- **Full REDUCE_AXIS gradient**: Implemented proper gradient for both `ADD` (sum) and `MAX` reductions through partial-axis reductions. Sum gradient expands the incoming gradient back to `src_shape` via reshape+expand. Max gradient uses an indicator mask (`x == max_expanded`) to route gradient only to argmax positions.
- **Better backward error**: `Tensor.backward` now gives a descriptive `failwith` message instead of `assert` when loss is not scalar.
- **Partial-axis reduction gradient test**: Verified `d/dx sum(x*x, axis=1)` for `[2;3]` tensor produces correct `2*x` gradients for all 6 elements.
- Total tests: 96 unit + 139 e2e = 235 all passing.

## Claude round 13 decisions

- **Max reduction gradient tie-splitting**: Fixed `REDUCE_AXIS MAX` gradient to normalize by tie count. When multiple elements share the maximum value, gradient is split equally among them (e.g., `[3,7,7,2]` → `[0, 0.5, 0.5, 0]` instead of `[0, 1, 1, 0]`). Uses `tie_count = sum(mask)` per reduced slice.
- **Buffer broadcast indexing in kernels**: Fixed a critical bug where realized buffers smaller than the kernel's iteration space were indexed out-of-bounds. Now uses broadcast-aware indexing: scalar buffers use index 0, partial reduction results (size = output_numel) use the outer loop variable, and other mismatched sizes use `idx / (numel / buf_size)`. Affects both elementwise and reduction kernels.
- **New gradient rules**: Added PAD (shrink back to original region), SHRINK (pad with zeros), FLIP (self-inverse), TRUNC (zero — non-differentiable).
- **Max reduction backward tests**: Full-reduction max gradient (no ties), tie-splitting test (`[3,7,7,2]` verifying `[0, 0.5, 0.5, 0]`), and partial-axis max gradient (`max(axis=1)` on `[2;3]` tensor).
- **Unary gradient tests**: `d/dx sum(sqrt(x))` and `d/dx sum(exp2(x))` with analytical verification.
- Total tests: 96 unit + 161 e2e = 257 all passing.

## Claude round 14 decisions

- **Stride-based broadcast indexing**: Replaced the fragile ratio-based `idx / (numel / buf_size)` heuristic with proper stride-based coordinate decomposition via `broadcast_index`. For each dimension, decomposes the flat iteration index into per-axis coordinates and maps them through the buffer's strides, zeroing out broadcast dimensions (where `buf_shape[d] = 1`). This correctly handles both leading-axis broadcasts (`[1;N]` → `[M;N]`) and trailing-axis broadcasts (`[M;1]` → `[M;N]`).
- **Shape metadata for realized buffers**: Added `realized_shapes` table alongside `realized_buffers` in the scheduler. Reduction outputs and elementwise kernel outputs now store their shape when realized, enabling correct broadcast index computation in downstream kernels.
- **Chained reduction broadcast test**: `loss = sum(sum(x, axis=1)^2)` on `[2;3]` tensor exercises the full pipeline: partial reduction → elementwise square (with broadcast of [2;1] result to [2;3] in the gradient) → full reduction → backward. Verifies `d/dx = 2*sum(x, axis=1)` broadcast correctly to `[[12,12,12],[30,30,30]]`.
- **Sin gradient test**: `d/dx sum(sin(x))` at `[0, pi/4, pi/2]` verifying `cos(x) = [1, sqrt(2)/2, 0]`.
- Total tests: 96 unit + 171 e2e = 267 all passing.

## Claude round 15 decisions

- **Input buffer shape metadata in step-1 copyin**: Added `buffer_shapes` table in `schedule.ml` that stores the tensor shape alongside buffer data when `from_float_list` is called. In step-1 copyin, the shape is passed to `store_realized` so downstream kernels can use `broadcast_index` for input buffers, not just reduction outputs. Previously, input buffers fell back to the ratio heuristic if they needed broadcast indexing.
- **`broadcast_index` validity guard**: Added explicit checks that buf_shape rank doesn't exceed out_shape rank, and that each buffer dimension is either 1 (broadcast) or matches the output dimension. Raises `invalid_arg` with descriptive messages on mismatch.
- **Movement op pass-through in `rebuild_expr`**: Extended the scheduler's expression rebuilder to strip `PERMUTE`, `PAD`, `SHRINK`, `FLIP` movement ops alongside `RESHAPE`/`EXPAND`/`CONTIGUOUS`. This enables gradient tensors containing these ops (e.g., permute gradient) to be realized through the flat-buffer kernel pipeline.
- **Input buffer broadcast tests**: Added two tests exercising input buffers with broadcast indexing via the step-1 path: `[1;3]` expanded to `[2;3]` (trailing broadcast) and `[3;1]` expanded to `[3;2]` (leading broadcast). Both use the stride-based `broadcast_index` with shape metadata stored during copyin.
- **Log2 gradient test**: `d/dx sum(log2(x))` at `[1, 2, 4]` verifying `1/(x*ln2)`.
- **Permute gradient test**: `d/dx sum(permute(x, [1;0]))` on `[2;3]` tensor verifying all-ones gradient.
- Total tests: 96 unit + 195 e2e = 291 all passing.

## Claude round 16 decisions

- **Where + comparison e2e test**: `where(a < b, b, a)` = elementwise max, verifying `WHERE`, `CMPLT` ops work end-to-end through the scheduler/renderer/compiler pipeline.
- **Cast forward test**: `cast(a < b, float32)` verifying bool→float cast through the full pipeline.
- **Expand gradient test**: `d/dx sum(expand(x, [2;3]))` where `x` is `[1;3]`, verifying gradient is `[2, 2, 2]` (each element contributes twice via expansion). Exercises the `EXPAND` gradient rule (reduce_sum over expanded dims).
- **Where gradient test (ReLU)**: `relu(x) = where(x > 0, x, 0)`, `d/dx sum(relu(x))` verifying gradient is `[0, 1, 0, 1]`. Exercises the `WHERE` gradient rule through the full backward pipeline.
- **Expand + mul forward test**: `expand([1;3] → [2;3]) * [2;3]` verifying broadcast elementwise computation works correctly.
- **Documented movement op flattening**: Added comment in `rebuild_expr` explaining that PERMUTE/PAD/SHRINK/FLIP pass-through is correct for gradient UOp graphs over flat realized buffers, but would need proper index remapping for general forward computation.
- Total tests: 96 unit + 221 e2e = 317 all passing.

## Claude round 17 decisions

- **Matmul operation**: Implemented `Tensor.matmul` as decomposed primitive ops matching tinygrad's approach: `a[N,K] @ b[K,M]` → reshape a to `[N,K,1]`, expand to `[N,K,M]`, reshape b to `[1,K,M]`, expand to `[N,K,M]`, elementwise multiply, sum over K (axis=1), reshape to `[N,M]`. This works end-to-end through the existing scheduler/renderer without any new kernel types.
- **Effective broadcast shape inference**: Added `infer_buffer_effective_shapes` that walks the UOp graph to determine each buffer's effective shape in the kernel's coordinate space. This is essential for matmul where two same-sized buffers (e.g., both `[2,2]`) have different broadcast patterns (`a` expands as `[2,2,1]` while `b` expands as `[1,2,2]`). Without this, the right-alignment heuristic in `broadcast_index` would give incorrect indexing.
- **Broadcast shape validation**: Added `is_valid_broadcast` checks before calling `broadcast_index`. If the inferred effective shape is incompatible with the kernel's output shape (can happen when a buffer appears in multiple contexts in the DAG), gracefully falls back to the realized shape or ratio heuristic.
- **Linear layer training test**: End-to-end test training `y = x @ w` via SGD for 100 steps. Learns `w ≈ [2.0, 3.0]` from `x = [[1,0],[0,1],[1,1]]`, `target = [[2],[3],[5]]`. Demonstrates the full forward → matmul → loss → backward → gradient → parameter update pipeline.
- **Matmul tests**: Forward (2x2), non-square (1x3 @ 3x1, 2x3 @ 3x2), gradient (`d/da sum(a@b)`, `d/db sum(a@b)` with analytical verification).
- Total tests: 96 unit + 246 e2e = 342 all passing.

## Claude round 18 decisions

- **Fixed backward after realization**: Added `lazy_uop` field to `Tensor.t` that preserves the original computation graph when `realize` replaces `uop` with a BUFFER node. `backward` now uses `lazy_uop` to differentiate through the computation graph even after tensors have been realized via `to_float_list`. Previously, calling `backward` after `to_float_list` would fail because the computation graph was lost.
- **ReLU helper**: Added `Tensor.relu` as `where(x > 0, x, 0)` using comparison + ternary select. Forward and gradient tests verify correct behavior for mixed positive/negative inputs.
- **Two-layer MLP training on XOR**: End-to-end test training a two-layer MLP (`relu(x @ w1) @ w2`) on the XOR problem via 200 SGD steps. Achieves loss ≈ 0.00002 with predictions [0.000, 0.991, 1.001, 0.001] matching targets [0, 1, 1, 0]. This is the strongest integration test — exercises matmul, relu, mean loss, backward through relu(matmul), and multi-step parameter updates.
- **Extended CPU exec**: Added 6, 7, 8-buffer dispatch cases to handle the larger kernel signatures generated by MLP backward passes.
- **Matmul contract**: Tightened to require exactly 2-D inputs with clear error messages (addressing codex review feedback about batch support claim).
- **Same-buffer dual-expand test**: Verifies correct broadcast indexing when two different-shaped buffers go through different reshape+expand paths to the same kernel output shape (addresses codex aliasing concern).
- Total tests: 96 unit + 271 e2e = 367 all passing.

## Claude round 19 decisions

- **Fixed lazy_uop leakage (codex review feedback)**: All tensor construction ops (`binop`, `unop`, `where_`, `cast`, `reshape`, `expand`, `permute`, `pad`, `shrink`, `flip`, `reduce`, `contiguous`) now explicitly set `lazy_uop = None` to prevent stale computation graphs from leaking through `{ a with ... }` record copies.
- **Per-path broadcast indexing**: Replaced global `infer_buffer_effective_shapes` (one shape per buffer ID) with per-path effective shape tracking in `rebuild_expr`. When the same buffer is accessed through different RESHAPE→EXPAND paths, each path gets its own broadcast index. EXPAND nodes capture their child RESHAPE's shape; inner RESHAPEs don't override if an outer context has already set the effective shape. This fixes the same-buffer aliasing bug where both paths got identical column-broadcast indexing.
- **Graph splicing for backward through realized tensors**: Added `lazy_graph_map` that maps realized BUFFER UOp IDs to their original computation graphs. `backward` now splices original computation graphs back into the loss expression, enabling differentiation through chains like: compute → realize → build new expression → backward. Previously, backward after `to_float_list` would fail because the computation graph was replaced by an opaque BUFFER.
- **Removed dead code**: Deleted `infer_buffer_effective_shapes` function (superseded by per-path tracking).
- **Reset hooks**: Added `Schedule.register_reset_hook` so `lazy_graph_map` is cleared on `Schedule.reset()`.
- **New regression tests**: backward-after-realize, same-buffer aliasing (outer sum with forward+backward), realize-reuse-backward.
- Total tests: 96 unit + 296 e2e = 392 all passing.

## Claude round 20 decisions

- **Robust EXPAND shape extraction (codex review feedback)**: EXPAND handler in `rebuild_expr` now walks through wrapper ops (CONTIGUOUS, CAST, PERMUTE, PAD, SHRINK, FLIP) to find the nearest RESHAPE, instead of only checking the immediate child. This prevents silent fallback to weaker heuristics when movement/cast ops are inserted between EXPAND and RESHAPE.
- **Metal GPU end-to-end execution**: Verified and tested the full Metal backend pipeline: Tensor API → Schedule → CStyle MSL render → Metal.Library compile → ComputePipelineState dispatch → shared buffer copyout. Added comprehensive Metal tests:
  - Metal reduction (sum, mean, max on GPU)
  - Metal matmul (2x2 @ 2x2 on GPU)
  - Metal backward (d/dx sum(x^2) = 2x, gradient computation on GPU)
  - Metal linear training (y=2x regression, 100 SGD steps, converges to w=2.000000 on GPU)
- **Float32 precision characterization**: Discovered that CPU and Metal produce identical forward/backward results per-step, but tiny float32 rounding differences compound over many iterations on chaotic loss surfaces (XOR MLP). The XOR problem at lr=0.1 diverges on Metal around step 120 while converging on CPU. Linear regression converges identically on both backends. This is expected behavior — not a bug.
- Total tests: 96 unit + 311 e2e = 407 all passing.

## Claude round 21 decisions

- **Structural `view_wrapper` predicate (codex review feedback)**: Added `Ops.Group.view_wrapper` list and `is_view_wrapper` predicate to `lib/ops.ml`. This replaces hardcoded op lists in the scheduler with a semantic predicate for single-source view/wrapper ops (RESHAPE, EXPAND, PERMUTE, PAD, SHRINK, FLIP, CAST, CONTIGUOUS). Used in `find_reshape` (EXPAND shape walker) and the new `is_path_dependent` helper.
- **Consolidated path-dependent cache bypass**: Extracted `is_path_dependent` helper in `rebuild_expr` that uses `is_view_wrapper` for movement ops, with explicit CAST exclusion (CAST resets eff_shape, so its result doesn't depend on the incoming path context). Replaces duplicated hardcoded op lists at cache lookup and cache store sites.
- **PERMUTE forward execution**: Implemented proper PERMUTE index transformation in the scheduler. When `rebuild_expr` encounters a PERMUTE node, it:
  1. Finds the child's shape from the nearest RESHAPE via `find_child_shape` walker
  2. Computes the permuted shape from child_shape and axes
  3. Generates a `permute_index` UOp expression that decomposes the flat output index into multi-dim coordinates, applies the inverse permutation, and recomposes into a flat input index
  4. Threads the transformed index via new `~cur_idx` parameter through `rebuild`
  This enables correct element reordering for transpose and general permutations. Added `permute_index` helper function and threaded `cur_idx` through all `rebuild` recursive calls.
- **New test**: `test_permute_forward` verifies that [2,3] transposed to [3,2] produces correct element order [1,4,2,5,3,6].
- Total tests: 96 unit + 318 e2e = 414 all passing.

## Claude round 22 decisions

- **CAST caching fix (codex review feedback)**: CAST now treated as path-dependent in `is_path_dependent`. Previously CAST was cached, but since it passes `cur_idx` to children, caching a CAST reached via different index paths (e.g., one through PERMUTE, one not) would bind the wrong index expression. Correctness over performance.
- **PERMUTE shape inference fix (codex review feedback)**: `find_child_shape` in the PERMUTE handler now stops at EXPAND nodes instead of walking through them. EXPAND changes the logical shape (adding dimensions), so the pre-EXPAND RESHAPE shape is wrong for computing the permutation mapping. EXPAND's own shape arg is used instead.
- **Broadcast indexing coordinate frame fix (codex review feedback)**: Threaded `idx_shape` alongside `cur_idx` through `rebuild`. After a PERMUTE transforms the index, `idx_shape` is updated to the unpermuted (child) shape so that `make_load → broadcast_index` decomposes the index in the correct coordinate frame. Previously, `output_shape` (the kernel's final shape) was always used, which would be wrong after a PERMUTE. Removed `numel` parameter from `rebuild_expr` since it's now computed dynamically from `idx_shape`.
- **New test**: `test_permute_broadcast` — permute([2,3]→[3,2]) + broadcast([1,2]→[3,2]) verifies correct composition of index transformation and broadcast indexing.
- Total tests: 96 unit + 325 e2e = 421 all passing.

## Claude round 23 decisions

- **PAD/SHRINK forward execution (codex review feedback)**: Implemented proper index transformations for PAD and SHRINK movement ops in `rebuild_expr`. Previously both were pass-through (no-op), now they correctly transform indices:
  - **SHRINK**: `shrink_index` decomposes flat index into multi-dim coords in the shrunk shape, adds lower-bound offsets, recomposes into flat index in the original shape. Uses `find_shape` walker to get original child shape.
  - **PAD**: `pad_index` decomposes flat index into padded-shape coords, checks bounds for each dimension (before padding, after padding), produces `(inner_idx, mask)`. The mask is 0.0 in padding regions, 1.0 in valid regions. The scheduler multiplies the loaded value by the mask to zero out padding.
- **PAD/SHRINK shape in PERMUTE's find_child_shape**: Updated to compute shapes through PAD (add before+after) and SHRINK (hi-lo), rather than walking through them as identity wrappers. This ensures PERMUTE's index remapping uses the correct post-PAD/post-SHRINK shape.
- **New tests**: `test_shrink_forward` (1D [4]→[2] and 2D [3,4]→[2,2]) and `test_pad_forward` (1D [3]→[6] and 2D [2,2]→[3,3]) verify element extraction and zero-padding.
- Total tests: 96 unit + 350 e2e = 446 all passing.

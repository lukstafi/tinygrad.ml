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

Port tinygrad: ~/tinygrad/ to OCaml. Where reasonable, minimize how much of the frontend niceties are ported if functionality is not significantly reduced. Port a CPU C backend and the CUDA and Metal MSL backends using the `cudajit` ~/ocaml-cudajit/ and `metal` ~/ocaml-metal/ packages, do not port the other lower-level backends. Balance faithfulness to the tinygrad source code with educational value and using idiomatic OCaml rather than trying to emulate Python semantics.

## Codex round 1 decisions

- Chosen strategy: middle-layer-first vertical slice with a minimal lazy tensor wrapper.
- Frontend intentionally minimized to keep focus on execution stack (`add`, `mul`, `sum`, `realize`) instead of Python niceties.
- Backends limited exactly to requested set: CPU C, CUDA via `cudajit`, Metal via `metal`.
- Scope reduced to 1D buffers for first runnable baseline; architecture leaves room to extend shapes/ops incrementally.
- Renderers kept explicit and readable for educational value rather than exhaustive tinygrad feature parity in round 1.

## Codex round 2 decisions

- Addressed major feedback by moving from pairwise binop execution to fused expression execution (`Uop.expr`) at realize time.
- Added per-device tensor cache entries (list keyed by device), replacing single-slot cache.
- Added additional ops (`sub`, `neg`, `sqrt`, `reciprocal`) to improve educational/functional value without exploding frontend scope.
- Added CPU kernel compilation cache keyed by expression digest to avoid recompiling equivalent fused kernels.
- Kept CUDA/Metal default stubs for local buildability; real backends remain in `experimental/` pending optional dependency wiring in dune.

## Codex round 3 decisions

- Integrated real Metal backend into default `src/` path (`src/metal_backend.ml`) with `run_expr` support for fused expression kernels.
- Updated dune library deps to include `metal` (`src/dune`) so runtime dispatch can target CPU and Metal in the main build.
- Kept CUDA as stub in this branch due unresolved `cudajit` dune package availability in this environment.
- Verified CPU build/test path remains green after Metal integration; Metal execution was not fully verified here due local runtime blocking during manual demo invocation.

## Codex round 4 decisions

- Added dedicated Metal runtime tests in `test/test_metal.ml` to validate fused binary and unary expression execution on the GPU backend.
- Wired Metal tests into `test/dune` so `dune test` now exercises both CPU and Metal execution paths.
- Tightened Metal kernel ABI semantics by switching the generated length arg in MSL from `constant int &n` to `constant int *n`, matching host-side `set_bytes` usage.
- Kept CUDA backend as a stub for now because `cudajit` libraries are present in local source checkout but not installed in this switch (`ocamlfind list` does not expose `cudajit.cuda` / `cudajit.nvrtc`).

## Codex round 5 decisions

- Implemented fused-expression CUDA execution path in `experimental/cuda_backend_real.ml` (NVRTC compile, CUDA module/function cache, H2D/D2H transfers, kernel launch).
- Kept `src/cuda_backend.ml` as the default stub in this environment because `cudajit` libraries are not available in the current switch; real backend is staged for optional wiring where those libs exist.
- Added CUDA runtime tests in `test/test_cuda.ml` with runtime skip when CUDA is unavailable.
- Extended Metal coverage with a chained-realize test in `test/test_metal.ml` to validate multi-kernel execution with intermediate realized tensors.

## Codex round 6 decisions

- Added backend-level reduction execution API (`Runtime.run_reduce`) with two reduction ops: `sum` and `max`.
- Implemented compiled reduction kernels in CPU C and Metal backends:
  - CPU: C kernel codegen and execution via existing shared-library path.
  - Metal: dedicated MSL reduction kernel (single-thread loop for correctness-first behavior).
- Updated Tensor reductions to use compiled backend reduction kernels (`sum`, `max`, `mean`) instead of host-side `to_array` folding.
- Added reduction coverage to tests:
  - `test/test_cpu.ml`: CPU `sum`/`max`/`mean`.
  - `test/test_metal.ml`: Metal `sum`/`max`/`mean` (runtime-skipped if Metal unavailable).

## Codex round 7 decisions

- Upgraded staged CUDA backend (`experimental/cuda_backend_real.ml`) to execute reductions on GPU with dedicated CUDA reduction kernels (`run_reduce`), instead of reducing on host after `run_expr`.
- Added CUDA reduction kernel rendering (`src/cuda_renderer.ml`: `render_reduce_kernel`) with expression-keyed kernel caching shared through existing compile path.
- Extended CUDA tests (`test/test_cuda.ml`) with `sum`/`max`/`mean` checks (automatically skipped when CUDA backend is unavailable in current environment).

## Codex round 8 decisions

- Added fused-reduction coverage on CPU (`test/test_cpu.ml`): validates `sum` and `max` over computed expressions (not only raw input tensors).
- Added cross-backend fused-reduction consistency checks:
  - Metal vs CPU (`test/test_metal.ml`)
  - CUDA vs CPU (`test/test_cuda.ml`, runtime-skipped when CUDA unavailable).
- Tightened CPU test tolerance to relative form (`max(1e-6, 1e-6 * abs(expected))`) for numeric stability parity with CUDA tests.

## Codex round 9 decisions

- Added Tensor `reshape` support (metadata view with numel check) and relaxed backend input validation to allow shape-mismatched-but-numel-equal inputs for reshape-compatible elementwise execution.
- Added Tensor-layer axis reductions returning tensors:
  - `sum_axis ~axes`
  - `max_axis ~axes`
  - `mean_axis ~axes`
  implemented as host-side reductions over realized data for correctness and broad axis support.
- Added tests for reshape + axis reductions and backend consistency:
  - CPU: reshape elementwise + `sum_axis`/`max_axis`/`mean_axis`
  - Metal/CUDA: axis-reduction consistency checks against CPU (CUDA path runtime-skipped when unavailable).

## Codex round 10 decisions

- Improved reshape semantics in `Tensor` by preserving realized per-device cache across reshape (with remapped cached buffer shapes), avoiding unnecessary re-realization after metadata-only view changes.
- Optimized `mean_axis` host reduction path to avoid a second `to_array` realization pass by scaling reduction output in-memory before wrapping as tensor data.
- Extended axis-reduction correctness tests:
  - CPU non-contiguous multi-axis checks (`axes=[0;2]` on `[2;3;4]`) for `sum_axis`, `max_axis`, `mean_axis`.
  - Metal/CUDA non-contiguous `sum_axis` consistency checks against CPU.
- Harmonized Metal test tolerance with relative epsilon style used in CPU/CUDA tests.

## Codex round 11 decisions

- Replaced ad-hoc reshape behavior with an explicit `Reshape` node in `Tensor` AST so shape views are represented structurally, including for unrealized expression graphs.
- Implemented shape-aware lowering (`lower_to_expr_with_shape`) that propagates requested output shape through `Reshape` nodes and emits input buffer views with matching shape metadata.
- Restored strict backend input-shape validation (removed prior permissive “numel-only” acceptance) now that reshape compatibility is handled during lowering rather than backend validation.
- Added CPU regression coverage for reshaped unrealized expressions (`reshape(add(a,b), [4;2])`) to validate the new lowering path end-to-end.

## Codex round 12 decisions

- Added a minimal reverse-mode autograd API to `Tensor`:
  - `Tensor.backward ?grad ~wrt output` computes gradients of `output` with respect to selected tensors.
  - Defaults to `ones_like(output)` upstream gradient (vector-Jacobian with an all-ones vector), which matches `sum(output)` gradients for elementwise outputs.
- Implemented gradient rules for the current AST surface:
  - `Binop`: `Add`, `Sub`, `Mul`
  - `Unop`: `Neg`, `Sqrt`, `Reciprocal`
  - `Reshape` (gradient reshaped back to source shape)
- Added shape-aware tensor constants (`full_with_shape`, `zeros_like`, `ones_like`) used by gradient construction.
- Added CPU autograd tests:
  - `d/dx sum(x*x) = 2x`
  - `d/da sum(a*b)=b` and `d/db sum(a*b)=a`
  - Simple SGD loop minimizing `sum((x-target)^2)` from `[0,0]` toward `[3,5]`.
- Fixed a backend-independent codegen bug exposed by autograd constants:
  - C/CUDA/Metal expression renderers now emit valid float literals (`1.0f` instead of invalid `1f`).

## Codex round 13 decisions

- Added lazy axis-reduction nodes to Tensor AST:
  - New `Reduce_axis` node carries `op`, normalized `axes`, and source tensor.
  - `sum_axis` / `max_axis` now keep reductions in the graph when `?device` is omitted.
  - Preserved previous eager behavior for explicit `?device` calls to keep backend-forward tests stable.
- Extended realization/lowering to handle reduction nodes:
  - `realize_result` can realize `Reduce_axis` nodes directly via host reduction path.
  - Expression lowering treats `Reduce_axis` children as realized input buffers when reductions appear inside larger elementwise expressions.
- Implemented reverse-mode gradients through axis reductions:
  - `sum_axis` backward: broadcast upstream gradient back to source shape.
  - `max_axis` backward: mask-based gradient routing to argmax positions.
  - `mean_axis` backward now works via existing chain rule (`sum_axis * constant`).
- Added CPU autograd coverage for reduction paths:
  - `d/dx sum_axis(x*x, axis=1) = 2x`
  - `d/dx max_axis(x, axis=1)` mask for unique maxima
  - `d/dx mean_axis(x, axis=1)` equal split factor (`1/axis_size`).

## Codex round 14 decisions

- Fixed `max_axis` backward tie handling in Tensor autograd:
  - `max_reduce_grad` now counts argmax ties per reduced slice and divides upstream gradient by tie count.
  - Preserves gradient mass for tied maxima (e.g., `[3,7,7,2]` now gives `[0,0.5,0.5,0]`).
- Added stronger CPU reduction-autograd tests:
  - Tie-splitting regression for `max_axis` backward.
  - Intermediate reduction expression case (`add(sum_axis(x), 1)`) to exercise `Reduce_axis` as a non-root node in forward/backward paths.
- Kept approach intentionally AST-host-centric:
  - Reduction backward remains implemented in host OCaml index space for clarity and educational value, instead of compiling extra reduction-gradient kernels.

## Codex round 15 decisions

- Added unary transcendental ops across the expression stack:
  - `Uop.unop`: added `Exp2`, `Log2`, `Sin`.
  - Tensor API: added `Tensor.exp2`, `Tensor.log2`, `Tensor.sin`.
  - Renderers: CPU/CUDA/Metal codegen now emits `exp2*`, `log2*`, `sin*` intrinsics.
- Extended autograd rules for new unary ops:
  - `d/dx exp2(x) = exp2(x) * ln(2)`
  - `d/dx log2(x) = 1 / (x * ln(2))`
  - `d/dx sin(x) = cos(x)` implemented as `sin(pi/2 - x)` using existing ops.
- Added CPU coverage for both forward and backward behavior:
  - forward numerics for `exp2`, `log2`, `sin`
  - gradient checks for `sum(exp2(x))`, `sum(log2(x))`, and `sum(sin(x))`.

## Codex round 16 decisions

- Added lazy `Expand` movement op to Tensor AST:
  - New `Expand` node with shape validation (`source dim` must be `1` or equal to target dim).
  - `Tensor.expand` API added for explicit broadcasting-style expansion.
- Implemented host realization for `Expand`:
  - Added stride-based `expand_host_data` to materialize expanded buffers correctly for arbitrary expanded axes.
  - Lowering treats `Expand` like `Reduce_axis` (realize and pass as kernel input), preserving current backend simplicity.
- Added `Expand` autograd rule:
  - Backward reduces upstream gradient over expanded axes (transpose of broadcast), then reshapes to source shape when needed.
- Added CPU tests:
  - Forward: `expand` values and `expand` used as intermediate node in elementwise expressions.
  - Backward: `d/dx sum(expand(x))` and `d/dx sum(expand(x)^2)`.

## Codex round 17 decisions

- Added lazy `Permute` movement op to Tensor AST:
  - New `Permute` node with axis validation (rank match, in-range, no duplicates).
  - `Tensor.permute` API added for axis reordering.
- Implemented host realization for `Permute`:
  - Added stride-based `permute_host_data` that remaps output coordinates back to source coordinates using inverse permutation.
  - Lowering handles `Permute` like `Expand`/`Reduce_axis` (realize first, then use as input buffer).
- Added `Permute` autograd rule:
  - Backward uses inverse permutation (`permute(upstream, inverse_axes)`).
- Added CPU tests:
  - Forward `permute` correctness on `[2;3] -> [3;2]`.
  - Backward `d/dx sum(permute(x)) = ones_like(x)`.

## Codex round 18 decisions

- Added `Tensor.matmul` using existing primitive tensor ops (`reshape`, `expand`, `mul`, `sum_axis`) instead of introducing backend-specific matmul kernels.
- Implemented matmul shape logic with broadcasted batch-prefix handling:
  - Supports rank-2+ operands.
  - Validates inner dimension compatibility (`K`) and right-aligned broadcastability of leading batch dims.
- Added CPU coverage for:
  - 2D matmul forward numerics.
  - Batched matmul with batch-prefix broadcasting.
  - Backward gradients for `sum(matmul(a,b))` wrt both operands.
- Strengthened permute-backward coverage with a weighted test (`sum(permute(x) * w)`) so gradient routing correctness depends on inverse-axis mapping, not just permutation-invariant sums.

## Codex round 19 decisions

- Added comparison ops to expression/UOp layer:
  - `Uop.binop` now includes `Lt`, `Eq`, `Ne` with renderer support across CPU/CUDA/Metal codegen paths.
- Added tensor-level conditional primitives:
  - `Tensor.lt`, `Tensor.eq`, `Tensor.ne`
  - `Tensor.where_` implemented via mask arithmetic (`cond*t + (1-cond)*f`)
  - `Tensor.relu` implemented as `where_(lt(0,x), x, 0)`.
- Added autograd behavior for comparisons:
  - Gradients for `Lt`/`Eq`/`Ne` are explicitly zero for both inputs (non-differentiable logical ops treated as stop-grad masks).
- Added CPU tests for new functionality:
  - comparison mask forward,
  - `where_` forward,
  - `relu` forward and backward mask behavior.
- Added an end-to-end matmul training loop regression in CPU tests (`y = x @ w`, MSE + SGD for 100 steps) to exercise multi-step forward/backward/update dynamics.

## Codex round 20 decisions

- Replaced mask-arithmetic `where_` lowering with a native ternary expression path:
  - Added `Uop.Where` in expression IR.
  - Wired ternary rendering in CPU/CUDA/Metal renderers using conditional select syntax.
  - Updated `Tensor.where_` to build an explicit `Where` node instead of expanding to `cond*t + (1-cond)*f`.
- Added explicit autograd rule for `Where`:
  - No gradient through condition (zero mask gradient).
  - Branch gradients routed with conditional masking:
    - `d/dt = where(cond, upstream, 0)`
    - `d/df = where(cond, 0, upstream)`.
- Added CPU regressions to target prior review risks:
  - `where` branch-selection test with `NaN` values in non-selected branches.
  - Nonlinear multi-step training test using `relu(matmul(...))` to ensure gradient flow through `where`/comparison/relu across optimization steps.
- Extended CPU C backend call dispatch arity support to 10 input buffers, which is needed for larger fused expressions produced by native `where` + training graphs.

## Codex round 21 decisions

- Added explicit `cast` support across tensor graph and expression IR:
  - New tensor node `Cast (dtype, x)` and expression node `Uop.Cast (dtype, x)`.
  - New `Tensor.cast ~dtype` API with support for `Dtype.F32`, `Dtype.I32`, and `Dtype.Bool`.
- Implemented backend renderer lowering for cast on CPU/CUDA/Metal:
  - `F32`: identity,
  - `I32`: truncating cast (via `int` conversion then back to float storage),
  - `Bool`: `0/1` mask from nonzero test.
- Added autograd rules for cast:
  - `cast -> F32` passes upstream gradient through unchanged.
  - `cast -> I32` and `cast -> Bool` are treated as non-differentiable (zero gradient).
- Added CPU regression coverage for cast forward/backward semantics:
  - Forward checks for `I32`, `Bool`, and `F32`.
  - Backward checks that only `cast(F32)` propagates gradient.

## Codex round 22 decisions

- Hardened numeric test assertions across CPU/CUDA/Metal tests:
  - Updated `check_close` to treat non-finite mismatches (`NaN`, `inf`) as failures, so branch-selection regressions cannot silently pass.
- Added explicit `where` backward routing test on CPU:
  - Verifies with custom upstream gradient that:
    - `d/dcond where(cond,t,f) = 0`,
    - `d/dt where(cond,t,f) = upstream` on true-mask positions,
    - `d/df where(cond,t,f) = upstream` on false-mask positions.
- Added Metal-vs-CPU parity coverage for new ops:
  - `where` with `NaN` in non-selected branches (ensures ternary select semantics),
  - `cast` parity for `I32`, `Bool`, and `F32`.

## Codex round 23 decisions

- Added direct Metal coverage for `where` backward routing:
  - New `test_where_backward_metal` checks gradients for `where(cond, t, f)` with custom upstream `[2,3,5,7]`.
  - Verifies expected masks:
    - `d/dcond = [0,0,0,0]`,
    - `d/dt = [2,0,5,0]`,
    - `d/df = [0,3,0,7]`.
- Test validates both devices in one place:
  - Asserts CPU gradient arrays first (sanity),
  - then asserts Metal gradient arrays against the same expected values.

## Codex round 24 decisions

- Extracted shared numeric test assertions into `test/test_helpers.ml`:
  - `check_close` with non-finite-aware behavior (`NaN`/`inf` mismatch fails),
  - `check_array` for elementwise array checks.
- Removed duplicated assertion code from:
  - `test/test_cpu.ml`,
  - `test/test_cuda.ml`,
  - `test/test_metal.ml`,
  and replaced it with `open Test_helpers`.
- This addresses cross-file maintenance risk for future assertion behavior changes by moving to a single source of truth.

## Codex round 25 decisions

- Added lazy movement ops `Pad` and `Shrink` to the Tensor AST in `src/tensor.ml`:
  - New APIs: `Tensor.pad : t -> (int * int) array -> t` and `Tensor.shrink : t -> (int * int) array -> t`.
  - Added rank/bounds validation with explicit error messages.
- Implemented host realization for both ops:
  - `pad_host_data`: maps padded output coordinates back into source and writes zeros in out-of-bounds regions.
  - `shrink_host_data`: slices source coordinates using `(lo, hi)` bounds.
  - `lower_to_expr_with_shape` treats `Pad`/`Shrink` like other movement ops (`Expand`/`Permute`): realize first, then feed as kernel inputs.
- Added autograd rules:
  - `d/dx sum(pad(x)) = shrink(upstream, bounds derived from padding)`.
  - `d/dx sum(shrink(x)) = pad(upstream, padding derived from source shape and bounds)`.
- Added CPU regressions in `test/test_cpu.ml`:
  - Forward coverage for `pad`, `shrink`, and composed `permute -> pad -> shrink`.
  - Backward coverage for both `pad` and `shrink`.
- Validation:
  - `dune exec test/test_cpu.exe` passes.
  - `dune test` passes (`test_cpu: ok`, `test_metal: ok`, CUDA skipped due unavailable backend).

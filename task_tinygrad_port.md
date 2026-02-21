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

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

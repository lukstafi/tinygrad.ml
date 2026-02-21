Port tinygrad: ~/tinygrad/ to OCaml. Where reasonable, minimize how much of the frontend niceties are ported if functionality is not significantly reduced. Port a CPU C backend and the CUDA and Metal MSL backends using the `cudajit` ~/ocaml-cudajit/ and `metal` ~/ocaml-metal/ packages, do not port the other lower-level backends. Balance faithfulness to the tinygrad source code with educational value and using idiomatic OCaml rather than trying to emulate Python semantics.

## Codex round 1 decisions

- Chosen strategy: middle-layer-first vertical slice with a minimal lazy tensor wrapper.
- Frontend intentionally minimized to keep focus on execution stack (`add`, `mul`, `sum`, `realize`) instead of Python niceties.
- Backends limited exactly to requested set: CPU C, CUDA via `cudajit`, Metal via `metal`.
- Scope reduced to 1D buffers for first runnable baseline; architecture leaves room to extend shapes/ops incrementally.
- Renderers kept explicit and readable for educational value rather than exhaustive tinygrad feature parity in round 1.

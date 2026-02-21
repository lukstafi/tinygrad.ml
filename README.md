# tinygrad.ml

An educational OCaml port of the tinygrad middle layer.

This implementation intentionally focuses on:
- lazy execution over a small tensor graph,
- kernel source rendering,
- compiled runtime dispatch,
- three backends only: CPU C, CUDA (`cudajit`), and Metal MSL (`metal`).

It intentionally does **not** try to reproduce Python frontend convenience APIs.

## Current scope

- Tensor rank: currently 1D buffers.
- Ops: elementwise `add`, `mul`, and `sum` reduction on realized output.
- Execution model: lazy graph nodes (`Data` / `Binop`) realized per selected backend.
- Backends:
  - CPU C: JIT compiles tiny C kernels with system `cc` and calls them via `ctypes`.
  - CUDA: generates CUDA C kernels, compiles with NVRTC (`cudajit.nvrtc`), launches with `cudajit.cuda`.
  - Metal: generates MSL kernels and dispatches with `metal`.

## Layout

- `src/uop.ml`: minimal UOp/binop representation.
- `src/program_spec.ml`: renderable kernel payload.
- `src/c_renderer.ml`, `src/cuda_renderer.ml`, `src/metal_renderer.ml`: backend kernel renderers.
- `src/cpu_c_backend.ml`, `src/cuda_backend.ml`, `src/metal_backend.ml`: execution backends.
- `src/runtime.ml`: device selection and backend dispatch.
- `src/tensor.ml`: minimal lazy tensor API.

## Build and test

```bash
dune build

dune test
```

Run the demo (device selected by `TG_DEVICE=cpu|cuda|metal`, default `cpu`):

```bash
TG_DEVICE=cpu dune exec tinygrad_ml_demo
```

## Notes

- CUDA/Metal execution depends on local toolchain and device availability.
- CPU backend requires a C compiler (`cc`) available on PATH.

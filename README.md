# tinygrad.ml

An educational OCaml port of the tinygrad middle layer.

This implementation focuses on:
- lazy expression graphs,
- fused kernel generation from those graphs,
- compiled runtime dispatch,
- three backends only: CPU C, CUDA (`cudajit`), and Metal MSL (`metal`).

It intentionally does **not** reproduce Python frontend convenience APIs.

## Current scope

- Tensor rank: flat buffers with shape metadata (supports reshape views and axis reductions in Tensor layer).
- Ops: elementwise `add`, `sub`, `mul`, `neg`, `sqrt`, `reciprocal`, `exp2`, `log2`, `sin`,
  compiled scalar reductions `sum`, `max`, `mean`,
  plus Tensor-layer axis reductions (`sum_axis`, `max_axis`, `mean_axis`), movement ops (`reshape`, `expand`, `permute`, `pad`, `shrink`, `flip`),
  and a minimal reverse-mode autograd API (`backward`) including axis-reduction gradients.
- Execution model:
  - tensors build lazy expression trees,
  - realization lowers a tree to `Uop.expr`,
  - renderer emits one fused kernel for the whole expression,
  - reduction ops emit dedicated fused reduction kernels.
- Cache behavior:
  - tensor realization cache is per-device,
  - CPU backend caches compiled kernels by expression key.
- Backends:
  - CPU C: active and tested (`cc` + `ctypes`/`dlopen`).
  - Metal: active and tested (`metal` package).
  - CUDA: default build uses a stub backend; a real `cudajit` implementation is provided in `experimental/cuda_backend_real.ml` for environments that wire in `cudajit.cuda` + `cudajit.nvrtc`.

## Layout

- `src/uop.ml`: expression IR (`Input`, `Const`, `Binop`, `Unop`).
- `src/tensor.ml`: minimal lazy tensor API + device-keyed cache.
- `src/c_renderer.ml`, `src/cuda_renderer.ml`, `src/metal_renderer.ml`: expression-to-kernel renderers.
- `src/cpu_c_backend.ml`: fused-kernel compile+execute with kernel caching.
- `src/cuda_backend.ml`: default CUDA stub backend.
- `experimental/cuda_backend_real.ml`: real `cudajit` CUDA backend implementation (staged for optional integration).
  - includes fused expression kernels and reduction kernel execution (`sum`/`max`) when wired in.
- `src/runtime.ml`: device selection and backend dispatch.

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

- CPU backend requires a C compiler (`cc`) on `PATH`.
- CUDA tests skip cleanly when no CUDA backend is available at runtime.

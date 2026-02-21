# tinygrad.ml

An educational OCaml port of the tinygrad middle layer.

This implementation focuses on:
- lazy expression graphs,
- fused kernel generation from those graphs,
- compiled runtime dispatch,
- three backends only: CPU C, CUDA (`cudajit`), and Metal MSL (`metal`).

It intentionally does **not** reproduce Python frontend convenience APIs.

## Current scope

- Tensor rank: currently 1D buffers.
- Ops: elementwise `add`, `sub`, `mul`, `neg`, `sqrt`, `reciprocal`, and `sum` reduction on realized output.
- Execution model:
  - tensors build lazy expression trees,
  - realization lowers a tree to `Uop.expr`,
  - renderer emits one fused kernel for the whole expression.
- Cache behavior:
  - tensor realization cache is per-device,
  - CPU backend caches compiled kernels by expression key.
- Backends:
  - CPU C: active and tested (`cc` + `ctypes`/`dlopen`).
  - CUDA/Metal: API stubs in default build; real implementations are staged in `experimental/` for environments with those packages available to dune.

## Layout

- `src/uop.ml`: expression IR (`Input`, `Const`, `Binop`, `Unop`).
- `src/tensor.ml`: minimal lazy tensor API + device-keyed cache.
- `src/c_renderer.ml`, `src/cuda_renderer.ml`, `src/metal_renderer.ml`: expression-to-kernel renderers.
- `src/cpu_c_backend.ml`: fused-kernel compile+execute with kernel caching.
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

- CUDA/Metal runtime integration code exists in `experimental/`, but this repository’s default dune setup keeps buildability in environments without those optional packages.
- CPU backend requires a C compiler (`cc`) on `PATH`.

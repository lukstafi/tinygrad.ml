Port tinygrad: ~/tinygrad/ to OCaml. Where reasonable, minimize how much of the frontend niceties are ported if functionality is not significantly reduced. Port a CPU C backend and the CUDA and Metal MSL backends using the `cudajit` ~/ocaml-cudajit/ and `metal` ~/ocaml-metal/ packages, do not port the other lower-level backends. Balance faithfulness to the tinygrad source code with educational value and using idiomatic OCaml rather than trying to emulate Python semantics.

## Claude round 2 decisions

- Added Metal GPU backend using `metal` OCaml package with shared-storage buffers, kernel caching, MSL compilation via Metal.Library, compute pipeline dispatch.
- Fixed Tensor data pipeline: `from_float_list` stores data via `Schedule.store_buffer_data`, `to_float_list` retrieves from realized Device.buffer via `Device.copyout_floats`.
- Implemented minimal Schedule: walks UOp graph to find unrealized BUFFER nodes, creates Copyin exec items that allocate device buffers and copy data.
- Implemented Realize: executes Copyin items (copies float data into device buffers) and Kernel items (renders, compiles, executes).
- CUDA backend not implemented (cudajit package not installed on this machine). Renderer generates valid CUDA code but no runtime execution.
- Metal backend uses loop-based kernels (1 threadgroup) since the scheduler doesn't yet emit SPECIAL ops for GPU thread indexing. Correct but not optimal.
- Extended CPU exec dispatch to handle 4-5 buffer arguments.

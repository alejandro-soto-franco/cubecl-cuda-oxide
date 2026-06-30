# cubecl-cuda-oxide

A [CubeCL](https://github.com/tracel-ai/cubecl) compiler backend that lowers
kernel definitions to PTX through the
[cuda-oxide](../cuda-oxide) LLVM-NVPTX pipeline instead of nvrtc. The result
is a pure-Rust GPU JIT: no C++ compiler, no nvrtc, no CUDA SDK headers at
build time.

## What it does

CubeCL's `Compiler` trait accepts a `KernelDefinition` (the CubeCL IR) and
must return a PTX representation. This crate walks the IR with
`compiler::walk`, maps it to a dialect-mir module, and drives
`mir_importer::compile_module_to_ptx` (the cuda-oxide LLVM-NVPTX pipeline) to
produce the final PTX bytes. The PTX is then loaded and launched through
`cudarc` exactly as the stock `cubecl-cuda` (nvrtc) path does.

Burn's `CubeBackend<CudaOxideRuntime>` is exposed as the public `CudaOxide`
type alias, so the backend satisfies Burn's trait bounds.

## M1 scope

Milestone 1 covers the irreducible elementwise-add "spine":

- Element type: `f32` only.
- Line size: 1 (scalar loads/stores; no vectorised `Line`).
- Operations: `Tensor::add` -- bounds-checked read/write, absolute-position
  index, PlainLayout metadata, info-buffer addressing.
- No fusion, no autotune, no tensor-core paths.

**Proven results (RTX 5060 Laptop, sm_120, N=4096 f32):**

- `cargo run --example tensor_add` prints `[11.0, 22.0, 33.0, 44.0]`.
- `cargo test --test parity`: cuda-oxide output is **bit-exact** against the
  stock nvrtc reference (zero tolerance, checked via `to_bits()` equality over
  4096 seeded values).
- JIT latency: cold ~192 ms (LLVM pipeline + PTX load), warm ~0.43 ms (cached
  kernel, at parity with nvrtc warm).

## Build contract

### Toolchain

The crate depends on `cuda-oxide`, which links `rustc_private` crates from the
nightly compiler. The nightly channel and exact revision are pinned in
`rust-toolchain.toml`; `rustup` picks it up automatically.

### Consumer requirement

Any binary, test, or example that transitively links `cuda-oxide`'s
`mir-importer` **must** declare at its crate root:

```rust
#![feature(rustc_private)]
extern crate rustc_driver;
```

Without these, linking fails with unresolved symbols from `rustc_private`.
Both declarations are already present in `examples/tensor_add.rs` and
`tests/parity.rs` as the canonical pattern to follow.

### Environment

Set `CUDA_OXIDE_TARGET` to the PTX target architecture before building or
running:

```sh
export CUDA_OXIDE_TARGET=sm_120   # RTX 50-series (Blackwell)
cargo run --example tensor_add
```

If the variable is unset, `mir_importer` defaults to a conservative `sm_80`.

## Repository layout

```
src/
  compiler/
    abi.rs     -- KernelDefinition -> dialect-mir parameter types + info-buffer layout
    walk.rs    -- CubeCL IR -> dialect-mir module walker
    mod.rs     -- CudaOxideCompiler (Compiler trait impl)
  compute/
    context.rs -- CUDA context: PTX load, kernel launch
    server.rs  -- CubeCL ComputeServer impl (vendored from cubecl-cuda)
    ...
  runtime.rs   -- CudaOxideRuntime (Runtime + CubeRuntime trait impls)
  lib.rs
examples/
  tensor_add.rs  -- end-to-end Tensor::add, asserts [11,22,33,44]
tests/
  parity.rs      -- bit-exact comparison vs checked-in nvrtc reference
  walk_table.rs  -- PTX-validity unit tests (no GPU required)
```

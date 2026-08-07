# cubecl-cuda-oxide

A [CubeCL](https://github.com/tracel-ai/cubecl) backend that compiles kernels to PTX through the [cuda-oxide](https://github.com/NVlabs/cuda-oxide) LLVM-NVPTX pipeline instead of nvrtc. Pure-Rust GPU JIT, needing no C++ compiler, nvrtc, or CUDA SDK headers at build time.

## Status

`burn` elementwise `add` runs end to end on the GPU through cuda-oxide PTX, bit-exact against the stock nvrtc backend.

Measured on RTX 5060 Laptop, sm_120, ptxas 13.1:

| Check | Result |
| --- | --- |
| `examples/tensor_add` | `[11, 22, 33, 44]` on device |
| `tests/parity` numeric parity vs nvrtc | bit-exact (`to_bits()` equality, zero tolerance), N = 4096 and N = 4097 |
| out-of-bounds bounds-check path (N = 4097, non-multiple of cube_dim) | bit-exact vs CPU oracle |
| `tests/walk_table` (5 cases) | every emitted PTX passes `ptxas -arch=sm_120` |
| `tests/parity` fixture provenance | reference cross-checked against CPU `a + b` and a live nvrtc run |
| JIT latency, cold (LLVM pipeline + PTX load) | cuda-oxide ~192 ms vs nvrtc ~34 ms |
| JIT latency, warm (cached kernel) | cuda-oxide ~0.43 ms vs nvrtc ~0.46 ms |

## How it works

CubeCL's `Compiler` trait hands a `KernelDefinition` (CubeCL IR) and expects a PTX representation. `CudaOxideCompiler` walks the IR into a `dialect-mir` function inside a `cuda_oxide_codegen::experimental::CodegenModule`, marks it as the kernel entry, and compiles it through `Compiler::compile_owned` for the PTX bytes. The PTX is loaded and launched through `cudarc`, reusing the `cubecl-cuda` server with the nvrtc step removed. `CudaOxide = CubeBackend<CudaOxideRuntime>` is the Burn backend.

## Progress

Working now:

- `f32` elementwise `add`, end to end, bit-exact against nvrtc.
- `dialect-mir` lowering of the add op surface: bounds-checked read and write, absolute-position index, PlainLayout metadata, info-buffer addressing. Line size 1 (scalar loads and stores).
- Fail-loud rejection of any unsupported op or address type at lowering.
- PTX load and launch through the vendored `cubecl-cuda` server.

Next:

1. More elementwise ops and reductions.
2. Vectorised `Line` (line size greater than 1) for the contiguous fast path.
3. Dtypes beyond `f32`; multi-type scalar metadata.
4. Non-contiguous and higher-rank tensors; `u64` metadata addressing.
5. Fusion and autotune; broader Burn model coverage.
6. Upstream the backend into a CubeCL fork.

## Build

`cuda-oxide-codegen`, `dialect-mir` and `dialect-nvvm` come from upstream [NVlabs/cuda-oxide](https://github.com/NVlabs/cuda-oxide) through `cuda_oxide_codegen::experimental`, as path dependencies on a local checkout. NVlabs is merging cuda-oxide into a `cuda-rust` mono-repo, which relocates these crates, so `Cargo.toml` avoids a git rev pin until that lands. `burn` is a path dependency at the pinned cubecl revision `1d628d7`, laid out as `../burn`.

All three cuda-oxide crates must come from the same checkout. Mixing them gives two distinct crate instances and a type mismatch at the pliron `Context` boundary, even at identical content. `pliron` itself is pinned to the git rev cuda-oxide's own workspace uses, not a registry version, for the same reason.

Toolchain is `stable`, pinned in `rust-toolchain.toml`; `rustup` selects it automatically.

Nothing here links `mir-importer`, so no crate root declares `#![feature(rustc_private)]` or `extern crate rustc_driver;`, and no toolchain needs to match a specific nightly's `rustc_driver`. `cuda_oxide_codegen::experimental::Compiler::discover` finds `llc`/`opt` on `PATH` (falling back to a versioned name such as `llc-22`), so a system LLVM 21+ install covers it and the `llvm-tools` rustup component is not required.

The standalone `Compiler` never hosts `rustc_driver`, so no build-time env knobs are needed. Set the PTX target before running:

```sh
export CUDA_OXIDE_TARGET=sm_120   # default is sm_120
cargo run --example tensor_add
```

## Layout

```
src/compiler/   CudaOxideCompiler: KernelDefinition -> dialect-mir -> PTX (abi, walk, mod)
src/compute/    CUDA context, PTX load and launch, ComputeServer (vendored from cubecl-cuda)
src/runtime.rs  CudaOxideRuntime, CudaOxide backend alias
examples/       tensor_add end to end
tests/          parity (bit-exact vs nvrtc), walk_table (PTX validity, no GPU)
```

## Licence

MIT OR Apache-2.0.

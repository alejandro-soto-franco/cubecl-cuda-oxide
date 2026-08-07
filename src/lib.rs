//! CubeCL backend that compiles kernels to PTX through cuda-oxide's
//! LLVM-NVPTX pipeline instead of nvrtc.
//!
//! M1 scope: a single elementwise op (`Tensor::add`), f32, `line_size = 1`,
//! no fusion, no autotune.

#![deny(clippy::all)]

extern crate alloc;
#[macro_use]
extern crate derive_new;

pub mod compiler;
pub mod compute;
pub mod device;
pub mod runtime;

pub use runtime::{CudaOxideDevice, CudaOxideRuntime};

/// Burn backend backed by the cuda-oxide PTX pipeline.
///
/// `CubeBackend<CudaOxideRuntime>` is single-parameter at this rev of
/// burn-cubecl (matches `pub type Cuda = CubeBackend<CudaRuntime>` in
/// burn-cuda), so no element-type type parameters are needed here.
pub type CudaOxide = burn_cubecl::CubeBackend<CudaOxideRuntime>;

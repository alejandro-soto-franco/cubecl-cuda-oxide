//! CubeCL backend that compiles kernels to PTX through cuda-oxide's
//! LLVM-NVPTX pipeline instead of nvrtc.
//!
//! M1 scope: a single elementwise op (`Tensor::add`), f32, `line_size = 1`,
//! no fusion, no autotune.

#![deny(clippy::all)]

pub mod compiler;

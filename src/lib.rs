//! CubeCL backend that compiles kernels to PTX through cuda-oxide's
//! LLVM-NVPTX pipeline instead of nvrtc.
//!
//! M1 scope: a single elementwise op (`Tensor::add`), f32, `line_size = 1`,
//! no fusion, no autotune.

#![feature(rustc_private)]
#![deny(clippy::all)]

extern crate alloc;
extern crate rustc_driver;
#[macro_use]
extern crate derive_new;

pub mod compiler;
pub mod compute;
pub mod device;
pub mod runtime;

//! Run a Burn elementwise `add` end to end on the GPU (RTX 5060 Laptop,
//! sm_120) through the pure-Rust cuda-oxide PTX path.
//!
//! Run with:
//! ```bash
//! CUDA_OXIDE_TARGET=sm_120 cargo run --example tensor_add
//! ```
//!
//! Expected output: `[11.0, 22.0, 33.0, 44.0]` with no assertion failure.
//!
//! ## Why the `Backend` trait and not `Tensor::<CudaOxide, 1>`
//!
//! `Tensor::<CudaOxide, 1>::from_data(..)` belongs to an older burn API. This
//! burn rev changed `Tensor` to `Tensor<const D: usize, K>` with no backend
//! type parameter: the backend is chosen at runtime by `burn-dispatch` through
//! a closed, compile-time `backend_list!` (Cpu/Cuda/Metal/Wgpu/...). An
//! out-of-tree backend such as `CudaOxide` cannot join that list without
//! forking burn. The faithful integration therefore drives the
//! `burn_backend::Backend` trait directly on the `CudaOxide` type itself:
//! `float_from_data` -> `float_add` -> `float_into_data`. That is the op path
//! `Tensor::add` would have lowered to, executed through cuda-oxide PTX.
//!
use burn_backend::ops::FloatTensorOps;
use burn_backend::{TensorData, read_sync};
use cubecl_cuda_oxide::{CudaOxide, CudaOxideDevice};

fn main() {
    // The compiler hardcodes `sm_120`; set the env var so the intent is explicit
    // and consistent with any future env-driven target override.
    // SAFETY: single-threaded, set before any backend/runtime work begins.
    unsafe {
        std::env::set_var("CUDA_OXIDE_TARGET", "sm_120");
    }

    let device = CudaOxideDevice::default();

    let a = <CudaOxide as FloatTensorOps<CudaOxide>>::float_from_data(
        TensorData::from([1.0f32, 2.0, 3.0, 4.0]),
        &device,
    );
    let b = <CudaOxide as FloatTensorOps<CudaOxide>>::float_from_data(
        TensorData::from([10.0f32, 20.0, 30.0, 40.0]),
        &device,
    );

    // The actual GPU launch: cuda-oxide PTX, loaded via cudarc, executed on sm_120.
    let c = CudaOxide::float_add(a, b);

    let data = read_sync(CudaOxide::float_into_data(c)).expect("read back failed");
    println!("{data:?}");

    data.assert_eq(&TensorData::from([11.0f32, 22.0, 33.0, 44.0]), true);
    println!("OK: Burn `float_add` ran on the GPU via cuda-oxide PTX (sm_120)");
}

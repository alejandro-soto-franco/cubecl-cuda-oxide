//! Device-service glue for [`CudaOxideServer`].
//!
//! This is the cuda-oxide analogue of `cubecl-cuda`'s `runtime.rs`, reduced to
//! the pieces required to satisfy the `ComputeServer` supertrait
//! [`DeviceService`]. The cubecl-cpp dependent paths (nvrtc compilation options,
//! WMMA/MMA feature probing, the C++ `CompilationOptions`) are dropped: PTX is
//! produced by [`crate::compiler::CudaOxideCompiler`], so device init only has to
//! build the CUDA context and the device-property table.
//!
//! M1 scope registers the standard scalar numeric types; richer feature probing
//! (tensor cores, TMA, fast-math variants) is left for later milestones.

use crate::compute::CudaOxideServer;
use crate::compute::context::CudaOxideContext;
use crate::device::{CUDA_MAX_BINDINGS, CudaDevice};
use cubecl_common::{
    device::{Device, DeviceId, DeviceService, ServerUtilitiesHandle},
    profile::TimingMethod,
};
use cubecl_core::{
    MemoryConfiguration,
    ir::{
        AddressType, DeviceProperties, ElemType, FloatKind, HardwareProperties, IntKind,
        MemoryDeviceProperties, Type, UIntKind, VectorSize,
        features::{AtomicUsage, TypeUsage},
    },
    server::ServerUtilities,
};
use cubecl_runtime::{allocator::PitchedMemoryLayoutPolicy, logging::ServerLogger};
use cudarc::driver::sys::cuDeviceTotalMem_v2;
use std::{mem::MaybeUninit, sync::Arc};

impl DeviceService for CudaOxideServer {
    fn init(device_id: DeviceId) -> Self {
        let device = CudaDevice::from_id(device_id);

        cudarc::driver::result::init().unwrap();
        let device_index = device.index as i32;
        let device_ptr = cudarc::driver::result::device::get(device_index).unwrap();

        // SAFETY: Calling CUDA driver FFI to query compute-capability attributes.
        // `device_ptr` is a valid device handle obtained from `device::get`.
        let arch_version = unsafe {
            let major = cudarc::driver::result::device::get_attribute(
                device_ptr,
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )
            .unwrap();
            let minor = cudarc::driver::result::device::get_attribute(
                device_ptr,
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            )
            .unwrap();
            major * 10 + minor
        } as u32;

        // Alignment returned by `cuMallocPitched`, considered optimal for row
        // alignment by CUDA and unchanged since at least the GTX 700 series.
        let mem_alignment = 512;

        // SAFETY: `device_ptr` is valid. `primary_ctx::retain` returns the primary
        // context, which is then set as current for the calling thread.
        let ctx = unsafe {
            let ctx = cudarc::driver::result::primary_ctx::retain(device_ptr).unwrap();
            cudarc::driver::result::ctx::set_current(ctx).unwrap();
            ctx
        };

        // SAFETY: `device_ptr` is valid. `cuDeviceTotalMem_v2` writes the total
        // device memory into the `MaybeUninit`, making `assume_init()` valid.
        let max_memory = unsafe {
            let mut bytes = MaybeUninit::uninit();
            cuDeviceTotalMem_v2(bytes.as_mut_ptr(), device_ptr);
            bytes.assume_init() as u64
        };
        let mem_properties = MemoryDeviceProperties {
            max_page_size: max_memory / 4,
            alignment: mem_alignment as u64,
        };

        // SAFETY: `device_ptr` is a valid CUDA device. All `get_attribute` calls
        // query read-only device properties via the CUDA driver API.
        let hardware_props = unsafe {
            use cudarc::driver::{result::device::get_attribute, sys::CUdevice_attribute::*};
            let warp_size =
                get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_WARP_SIZE).unwrap() as u32;
            let max_shared = get_attribute(
                device_ptr,
                CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )
            .unwrap() as usize;
            let max_threads = get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK)
                .unwrap() as u32;
            let block_dim_x =
                get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X).unwrap();
            let block_dim_y =
                get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y).unwrap();
            let block_dim_z =
                get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z).unwrap();
            let max_cube_dim = (block_dim_x as u32, block_dim_y as u32, block_dim_z as u32);

            let grid_dim_x = get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X).unwrap();
            let grid_dim_y = get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y).unwrap();
            let grid_dim_z = get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z).unwrap();
            let max_cube_count = (grid_dim_x as u32, grid_dim_y as u32, grid_dim_z as u32);

            let num_streaming_multiprocessors = Some(
                get_attribute(device_ptr, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT).unwrap() as u32,
            );

            HardwareProperties {
                load_width: 128,
                plane_size_min: warp_size,
                plane_size_max: warp_size,
                max_bindings: CUDA_MAX_BINDINGS,
                max_shared_memory_size: max_shared,
                max_cube_count,
                max_units_per_cube: max_threads,
                max_cube_dim,
                num_streaming_multiprocessors,
                num_tensor_cores: None,
                min_tensor_cores_dim: None,
                num_cpu_cores: None,
                max_vector_size: VectorSize::MAX,
                cube_mma_reserved_shared_memory: 0,
            }
        };

        let mut device_props = DeviceProperties::new(
            Default::default(),
            mem_properties.clone(),
            hardware_props,
            TimingMethod::System,
        );
        register_supported_types(&mut device_props);

        // Grid constants (`__grid_constant__`) are available from Volta onward and
        // change how kernel metadata is threaded to the launch in `server.rs`.
        let grid_constants = arch_version >= 70;

        let cuda_ctx = CudaOxideContext::new(grid_constants, device_props.clone(), ctx);
        let logger = Arc::new(ServerLogger::default());
        let policy = PitchedMemoryLayoutPolicy::new(device_props.memory.alignment as usize);
        let utilities = ServerUtilities::new(device_props, logger, (), policy);

        CudaOxideServer::new(
            cuda_ctx,
            mem_properties,
            MemoryConfiguration::default(),
            mem_alignment,
            device_id,
            utilities,
        )
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        self.utilities() as ServerUtilitiesHandle
    }
}

/// Registers the standard scalar numeric types supported by the CUDA backend.
///
/// This mirrors `cubecl-cpp`'s `register_supported_types` using only `cubecl-core`
/// types, so the crate carries no C++ compiler dependency.
fn register_supported_types(props: &mut DeviceProperties) {
    props.register_address_type(AddressType::U32);
    props.register_address_type(AddressType::U64);

    let supported_types = [
        ElemType::UInt(UIntKind::U8),
        ElemType::UInt(UIntKind::U16),
        ElemType::UInt(UIntKind::U32),
        ElemType::UInt(UIntKind::U64),
        ElemType::Int(IntKind::I8),
        ElemType::Int(IntKind::I16),
        ElemType::Int(IntKind::I32),
        ElemType::Int(IntKind::I64),
        ElemType::Float(FloatKind::BF16),
        ElemType::Float(FloatKind::F16),
        ElemType::Float(FloatKind::F32),
        ElemType::Float(FloatKind::Flex32),
        ElemType::Float(FloatKind::F64),
        ElemType::Bool,
    ];

    let supported_atomic_types = [
        ElemType::Int(IntKind::I32),
        ElemType::Int(IntKind::I64),
        ElemType::UInt(UIntKind::U32),
        ElemType::UInt(UIntKind::U64),
        ElemType::Float(FloatKind::F32),
    ];

    for ty in supported_types {
        props.register_type_usage(ty, TypeUsage::all());
    }

    for ty in supported_atomic_types {
        props.register_atomic_type_usage(Type::atomic(ty), AtomicUsage::Add | AtomicUsage::LoadStore);
    }
}

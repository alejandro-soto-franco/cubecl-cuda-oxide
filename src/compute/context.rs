use cubecl_common::backtrace::BackTrace;
use cubecl_runtime::{
    compiler::CompilationError,
    validation::{validate_cube_dim, validate_units},
};

use super::storage::gpu::GpuResource;
use crate::compiler::{CudaOxideCompiler, PtxRepresentation};
use crate::compute::stream::Stream;
use cubecl_common::compilation_cache::CompilationCache;
use cubecl_common::hash::StableHash;
use cubecl_core::{ir::DeviceProperties, prelude::*};
use cubecl_runtime::timestamp_profiler::TimestampProfiler;
use cubecl_runtime::{compiler::CubeTask, logging::ServerLogger};
use cudarc::driver::DriverError;
use cudarc::driver::sys::CUfunc_st;
use cudarc::driver::sys::{CUctx_st, CUfunction_attribute, CUtensorMap};
use std::collections::HashMap;
use std::ffi::CString;
use std::ffi::c_char;
use std::os::raw::c_void;
use std::sync::Arc;

use cubecl_common::cache::CacheOption;

#[derive(Debug)]
pub(crate) struct CudaOxideContext {
    pub context: *mut CUctx_st,
    pub module_names: HashMap<KernelId, CompiledKernel>,
    ptx_cache: Option<CompilationCache<StableHash, PtxCacheEntry>>,
    pub timestamps: TimestampProfiler,
    /// Whether the device supports grid constants (`__grid_constant__`), which
    /// the launch path uses to decide how kernel metadata is passed.
    pub grid_constants: bool,
    pub properties: DeviceProperties,
}

#[derive(Debug)]
pub struct CompiledKernel {
    cube_dim: CubeDim,
    shared_mem_bytes: usize,
    func: *mut CUfunc_st,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq, Clone)]
pub struct PtxCacheEntry {
    entrypoint_name: String,
    shared_mem_bytes: usize,
    ptx: Vec<std::ffi::c_char>,
}

impl CudaOxideContext {
    pub fn new(
        grid_constants: bool,
        properties: DeviceProperties,
        context: *mut CUctx_st,
    ) -> Self {
        Self {
            context,
            module_names: HashMap::new(),
            ptx_cache: {
                use cubecl_runtime::config::RuntimeConfig;
                let config = cubecl_runtime::config::CubeClRuntimeConfig::get();
                if let Some(cache) = &config.compilation.cache {
                    let root = cache.root();
                    Some(CompilationCache::new(
                        "ptx",
                        CacheOption::default().name("cuda").root(root),
                    ))
                } else {
                    None
                }
            },
            timestamps: TimestampProfiler::default(),
            grid_constants,
            properties,
        }
    }

    /// Switches the current CUDA context to this context.
    pub fn unsafe_set_current(&self) -> Result<(), DriverError> {
        // SAFETY: `self.context` is a valid CUDA context obtained from `primary_ctx::retain`
        // during server initialization and remains valid for the server's lifetime.
        unsafe { cudarc::driver::result::ctx::set_current(self.context) }
    }

    pub fn compile_kernel(
        &mut self,
        kernel_id: &KernelId,
        kernel: Box<dyn CubeTask<CudaOxideCompiler>>,
        mode: ExecutionMode,
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        let hash = if let Some(cache) = &self.ptx_cache {
            let hash = kernel_id.stable_hash();

            if let Some(entry) = cache.get(&hash) {
                log::trace!("Using PTX cache");

                self.load_ptx(
                    entry.ptx.clone(),
                    kernel_id.clone(),
                    entry.entrypoint_name.clone(),
                    kernel_id.cube_dim,
                    entry.shared_mem_bytes,
                )?;
                return Ok(());
            }
            Some(hash)
        } else {
            None
        };

        log::trace!("Compiling kernel");

        validate_cube_dim(&self.properties, kernel_id)?;
        validate_units(&self.properties, kernel_id)?;

        let kernel_compiled =
            kernel.compile(&mut Default::default(), &(), mode, kernel.address_type())?;

        let cube_dim = kernel_compiled.cube_dim;

        logger.log_compilation(&kernel_compiled);

        // The `CudaOxideCompiler` already produced PTX through the cuda-oxide
        // LLVM-NVPTX pipeline, so there is no nvrtc invocation here. Take the PTX
        // bytes from the compiled kernel's `PtxRepresentation` and ensure a
        // trailing NUL so the driver's `module::load_data` accepts the buffer.
        let repr: &PtxRepresentation =
            kernel_compiled
                .repr
                .as_ref()
                .ok_or_else(|| CompilationError::Generic {
                    reason: "compiled kernel is missing its PTX representation".to_string(),
                    backtrace: BackTrace::capture(),
                })?;
        let mut ptx: Vec<c_char> = repr.ptx.iter().map(|&b| b as c_char).collect();
        if ptx.last() != Some(&0) {
            ptx.push(0);
        }

        // M1 lowers no shared memory.
        let shared_mem_bytes = 0;

        // Single source of truth for the entry-point name: `repr.entry_name`,
        // which is set by `walk::entry_name(&kernel)` — the same function that
        // `lower_kernel` passes to `MirFuncOp::set_symbol_name`.  Using
        // `kernel_compiled.entrypoint_name` (= raw `options.kernel_name`) would
        // diverge when the name is blank or has surrounding whitespace, because
        // `walk::entry_name` trims and falls back to `"kernel"` while
        // `entrypoint_name` does not.
        let entry_name = repr.entry_name.clone();

        if let Some(cache) = &mut self.ptx_cache {
            let result = cache.insert(
                hash.unwrap(),
                PtxCacheEntry {
                    entrypoint_name: entry_name.clone(),
                    shared_mem_bytes,
                    ptx: ptx.clone(),
                },
            );
            if let Err(err) = result {
                log::warn!("Unable to save the ptx {err:?}");
            }
        }

        self.load_ptx(
            ptx,
            kernel_id.clone(),
            entry_name,
            cube_dim,
            shared_mem_bytes,
        )?;
        Ok(())
    }

    fn load_ptx(
        &mut self,
        ptx: Vec<c_char>,
        kernel_id: KernelId,
        entrypoint_name: String,
        cube_dim: CubeDim,
        shared_mem_bytes: usize,
    ) -> Result<(), CompilationError> {
        let func_name = CString::new(entrypoint_name).unwrap();
        // SAFETY: `ptx` is a valid null-terminated PTX binary from NVRTC. `func_name` is a
        // null-terminated `CString` matching the kernel entry point in the compiled module.
        let func = unsafe {
            let module = cudarc::driver::result::module::load_data(ptx.as_ptr() as *const _)
                .map_err(|err| CompilationError::Generic {
                    reason: format!("Unable to load the PTX: {err}"),
                    backtrace: BackTrace::capture(),
                })?;

            cudarc::driver::result::module::get_function(module, func_name).map_err(|err| {
                CompilationError::Generic {
                    reason: format!("Unable to fetch the function from the module: {err:?}"),
                    backtrace: BackTrace::capture(),
                }
            })?
        };

        self.module_names.insert(
            kernel_id.clone(),
            CompiledKernel {
                cube_dim,
                shared_mem_bytes,
                func,
            },
        );

        Ok(())
    }

    pub fn execute_task(
        &mut self,
        stream: &mut Stream,
        kernel_id: KernelId,
        dispatch_count: (u32, u32, u32),
        tensor_maps: &[CUtensorMap],
        resources: &[GpuResource],
        const_info: Option<*mut c_void>,
    ) -> Result<(), LaunchError> {
        let mut bindings = tensor_maps
            .iter()
            .map(|map| map as *const _ as *mut c_void)
            .collect::<Vec<_>>();
        bindings.extend(resources.iter().map(|memory| memory.binding));
        bindings.extend(const_info);

        let kernel = self.module_names.get(&kernel_id).unwrap();
        let cube_dim = kernel.cube_dim;
        // SAFETY: `kernel.func` is a valid function handle from a loaded module.
        // `stream.sys` is a valid CUDA stream. `bindings` contains valid device pointers
        // for all kernel arguments. The dispatch and cube dimensions are validated by
        // the caller.
        unsafe {
            cudarc::driver::result::function::set_function_attribute(
                kernel.func,
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                kernel.shared_mem_bytes as i32,
            )
            .map_err(|err| LaunchError::Unknown {
                reason: format!("{err}"),
                backtrace: BackTrace::capture(),
            })?;
            cudarc::driver::result::launch_kernel(
                kernel.func,
                dispatch_count,
                (cube_dim.x, cube_dim.y, cube_dim.z),
                // Shared memory is collected into a single buffer, with each shared memory being
                // an offset pointer
                kernel.shared_mem_bytes as u32,
                stream.sys,
                &mut bindings,
            )
            .map_err(|err| LaunchError::Unknown {
                reason: format!("{err}"),
                backtrace: BackTrace::capture(),
            })?;
        };

        Ok(())
    }
}

//! `CudaOxideCompiler`: a `cubecl_runtime::Compiler` that lowers a CubeCL
//! `KernelDefinition` to PTX through cuda-oxide's dialect-mir / LLVM-NVPTX
//! pipeline (`mir_importer::compile_module_to_ptx`), bypassing nvrtc.
//!
//! M1 scope is the irreducible elementwise-add "spine"; see [`walk`].

use core::fmt;

use cubecl_common::backtrace::BackTrace;
use cubecl_core::ir::{ElemType, StorageType};
use cubecl_runtime::{
    compiler::{CompilationError, Compiler},
    kernel::KernelDefinition,
    server::ExecutionMode,
};

pub mod abi;
pub mod walk;

/// PTX produced by the cuda-oxide pipeline. The CubeCL `Compiler::Representation`
/// bound is `Display`, which renders the PTX text.
#[derive(Debug, Clone)]
pub struct PtxRepresentation {
    /// Raw PTX bytes (UTF-8 assembly text).
    pub ptx: Vec<u8>,
    /// Name of the `.visible .entry` kernel.
    pub entry_name: String,
    /// Launch geometry carried over from the kernel definition.
    pub cube_dim: (u32, u32, u32),
}

impl fmt::Display for PtxRepresentation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&String::from_utf8_lossy(&self.ptx))
    }
}

/// CubeCL compiler backend targeting cuda-oxide's PTX pipeline.
#[derive(Debug, Clone, Default)]
pub struct CudaOxideCompiler;

impl Compiler for CudaOxideCompiler {
    type Representation = PtxRepresentation;
    /// M1: no compilation options.
    type CompilationOptions = ();

    fn compile(
        &mut self,
        kernel: KernelDefinition,
        _compilation_options: &Self::CompilationOptions,
        _mode: ExecutionMode,
        _addr_type: StorageType,
    ) -> Result<Self::Representation, CompilationError> {
        let mut ctx = pliron::context::Context::new();
        mir_importer::translator::register_dialects(&mut ctx);

        let module_op =
            walk::lower_kernel(&mut ctx, &kernel).map_err(|reason| CompilationError::Generic {
                reason,
                backtrace: BackTrace::capture(),
            })?;

        let cfg = mir_importer::PtxConfig::new("sm_120");
        let ptx = mir_importer::compile_module_to_ptx(&mut ctx, module_op, &cfg).map_err(|e| {
            CompilationError::Generic {
                reason: format!("cuda-oxide ptx pipeline failed: {e}"),
                backtrace: BackTrace::capture(),
            }
        })?;

        Ok(PtxRepresentation {
            ptx,
            entry_name: walk::entry_name(&kernel),
            cube_dim: (kernel.cube_dim.x, kernel.cube_dim.y, kernel.cube_dim.z),
        })
    }

    fn elem_size(&self, elem: ElemType) -> usize {
        elem.size()
    }

    fn extension(&self) -> &'static str {
        "ptx"
    }
}

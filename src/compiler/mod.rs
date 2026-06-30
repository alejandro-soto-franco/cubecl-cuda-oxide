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

#[cfg(test)]
mod tests {
    //! Entry-symbol invariant tests (no GPU required).
    //!
    //! The single source of truth is `walk::entry_name(&kernel)`:
    //! - `lower_kernel` passes it to `MirFuncOp::set_symbol_name` (PTX symbol).
    //! - `PtxRepresentation.entry_name` stores the same value.
    //! - `CudaOxideContext::compile_kernel` now uses `repr.entry_name` (not
    //!   `kernel_compiled.entrypoint_name`) when calling `get_function`.
    //!
    //! These tests verify all three are consistent without launching a kernel.
    use super::*;
    use cubecl_core::{
        ir::{ElemType, Scope, StorageType, UIntKind},
        server::ExecutionMode,
    };
    use cubecl_runtime::kernel::{KernelDefinition, KernelOptions};
    use cubecl_runtime::server::CubeDim;

    /// Build a minimal `KernelDefinition` with no buffers and an empty body so
    /// that `lower_kernel` succeeds without needing any GPU or CubeCL codegen.
    fn minimal_kernel(kernel_name: &str) -> KernelDefinition {
        KernelDefinition {
            buffers: vec![],
            tensor_maps: vec![],
            scalars: vec![],
            cube_dim: CubeDim { x: 1, y: 1, z: 1 },
            body: Scope::root(false),
            options: KernelOptions {
                kernel_name: kernel_name.to_string(),
                ..Default::default()
            },
        }
    }

    /// `walk::entry_name` trims whitespace and falls back to `"kernel"` when
    /// the name is blank.  These are the ONLY transformations applied to
    /// `kernel_name` before it becomes the PTX symbol and `repr.entry_name`.
    #[test]
    fn entry_name_trim_and_fallback() {
        assert_eq!(walk::entry_name(&minimal_kernel("")), "kernel");
        assert_eq!(walk::entry_name(&minimal_kernel("  ")), "kernel");
        assert_eq!(walk::entry_name(&minimal_kernel("my_kernel")), "my_kernel");
        assert_eq!(walk::entry_name(&minimal_kernel("  padded  ")), "padded");
    }

    /// The PTX emitted by the full cuda-oxide pipeline must contain
    /// `.visible .entry <repr.entry_name>`, and `repr.entry_name` must equal
    /// what `walk::entry_name` returns -- verifying the round-trip from
    /// `KernelDefinition` -> PTX symbol -> `get_function` name.
    #[test]
    fn ptx_entry_symbol_matches_repr_entry_name() {
        let kernel = minimal_kernel("test_oxide_kernel");
        let expected_name = walk::entry_name(&kernel);

        let mut compiler = CudaOxideCompiler;
        // addr_type is ignored by our compiler; any valid value works.
        let addr_ty = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let repr = compiler
            .compile(kernel, &(), ExecutionMode::Checked, addr_ty)
            .expect("minimal kernel should compile to PTX");

        // repr.entry_name must match walk::entry_name.
        assert_eq!(
            repr.entry_name, expected_name,
            "repr.entry_name diverged from walk::entry_name"
        );

        // PTX text must contain the entry declaration.
        let ptx = String::from_utf8_lossy(&repr.ptx);
        let expected_decl = format!(".visible .entry {}", repr.entry_name);
        assert!(
            ptx.contains(&expected_decl),
            "PTX does not contain expected entry declaration {expected_decl:?}\n\
             PTX:\n{ptx}"
        );
    }
}

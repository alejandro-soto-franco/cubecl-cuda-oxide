//! `CudaOxideCompiler`: a `cubecl_runtime::Compiler` that lowers a CubeCL
//! `KernelDefinition` to PTX through cuda-oxide's dialect-mir / LLVM-NVPTX
//! pipeline (`cuda_oxide_codegen::experimental`), bypassing nvrtc.
//!
//! M1 scope is the irreducible elementwise-add "spine"; see [`walk`].

use core::fmt;

use cubecl_common::backtrace::BackTrace;
use cubecl_core::ir::{ElemType, StorageType, UIntKind};
use cuda_oxide_codegen::experimental::{CodegenModule, CompileOptions, Compiler as OxideCompiler, Target};
use cubecl_runtime::{
    compiler::{CompilationError, Compiler},
    kernel::KernelDefinition,
    server::ExecutionMode,
};
use pliron::basic_block::BasicBlock;
use pliron::context::{Context, Ptr};
use pliron::linked_list::ContainsLinkedList;
use pliron::op::Op;

pub mod abi;
pub mod walk;

/// PTX target architecture, read from `CUDA_OXIDE_TARGET`. Defaults to
/// `sm_120`: the M1 spine, the parity fixtures, and `spine_add_lowers_and_validates`
/// all assume this target, and every measured result in the README is against
/// it. `sm_80` (a plausible-looking "lowest common denominator" default) would
/// silently retarget every one of those.
const DEFAULT_TARGET: &str = "sm_120";

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
        addr_type: StorageType,
    ) -> Result<Self::Representation, CompilationError> {
        // M1 hardwires a u32 info pointer and u32-unit metadata offsets.
        // Guard loudly against any other address width so callers discover the
        // limitation immediately rather than silently reading the wrong slot.
        let u32_addr = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        if addr_type != u32_addr {
            return Err(CompilationError::Generic {
                reason: format!(
                    "unsupported address type: {addr_type:?} (M1 supports U32 metadata addressing only)"
                ),
                backtrace: BackTrace::capture(),
            });
        }

        let entry_name = walk::entry_name(&kernel);

        let mut module = CodegenModule::new("kernel_module").map_err(|error| {
            CompilationError::Generic {
                reason: format!(
                    "cuda-oxide module creation failed ({:?}): {error}",
                    error.stage()
                ),
                backtrace: BackTrace::capture(),
            }
        })?;

        module
            .edit(|ctx, module_op| {
                let block = module_block(ctx, module_op);
                walk::lower_kernel(ctx, block, &kernel)
            })
            .map_err(|reason| CompilationError::Generic {
                reason,
                backtrace: BackTrace::capture(),
            })?;

        module
            .mark_kernel_entry(&entry_name)
            .map_err(|error| CompilationError::Generic {
                reason: format!(
                    "cuda-oxide kernel-entry marking failed ({:?}): {error}",
                    error.stage()
                ),
                backtrace: BackTrace::capture(),
            })?;

        // Read the target after a successful lowering, not before: a failed
        // lowering should report the lowering error, not an unrelated target
        // parse failure.
        let target_text =
            std::env::var("CUDA_OXIDE_TARGET").unwrap_or_else(|_| DEFAULT_TARGET.to_string());
        let target = Target::parse(&target_text).map_err(|error| CompilationError::Generic {
            reason: format!(
                "invalid CUDA_OXIDE_TARGET ({:?}): {error}",
                error.stage()
            ),
            backtrace: BackTrace::capture(),
        })?;
        let options = CompileOptions::new(target);

        let compiler = OxideCompiler::discover().map_err(|error| CompilationError::Generic {
            reason: format!(
                "cuda-oxide toolchain discovery failed ({:?}): {error}",
                error.stage()
            ),
            backtrace: BackTrace::capture(),
        })?;

        let ptx = compiler
            .compile_owned(module, &options)
            .map_err(|error| CompilationError::Generic {
                reason: format!(
                    "cuda-oxide ptx pipeline failed ({:?}): {error}",
                    error.stage()
                ),
                backtrace: BackTrace::capture(),
            })?
            .into_ptx();

        Ok(PtxRepresentation {
            ptx,
            entry_name,
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

/// The module's single top-level block, created if the module has none.
///
/// Upstream provides no such accessor; `mark_kernel_entry` requires exactly
/// one region and exactly one top-level block, so a caller building fresh IR
/// must set this up itself.
fn module_block(ctx: &mut Context, module: &impl Op) -> Ptr<BasicBlock> {
    let module_region = module.get_operation().deref(ctx).get_region(0);
    let existing = {
        let region = module_region.deref(ctx);
        region.iter(ctx).next()
    };
    match existing {
        Some(block) => block,
        None => {
            let block = BasicBlock::new(ctx, None, vec![]);
            block.insert_at_back(module_region, ctx);
            block
        }
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
        // M1 only accepts U32 addressing; pass the matching StorageType.
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

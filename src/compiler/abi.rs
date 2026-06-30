//! Kernel ABI: mapping CubeCL `KernelDefinition` buffers/scalars to the
//! dialect-mir function-parameter types the walk builds the entry block from.
//!
//! ## Parameter order (full `Tensor::add` surface)
//!
//! The kernel signature is, in order:
//!
//! 1. **Buffer pointers.** Each global buffer becomes one `MirPtrType` in the
//!    GLOBAL address space, in declaration order (`kernel.buffers[0]` -> param
//!    0, ...). The pointee is the buffer's element type (`f32` only for M1).
//!    Every buffer pointer is `is_mutable = true`: `KernelArg` carries no
//!    per-buffer visibility, so we take the writable superset, which still
//!    permits loads from read-only inputs.
//! 2. **Scalar values.** Each `ScalarKernelArg { ty: U32, count }` expands to
//!    `count` by-value `i32` (signless 32-bit) params. These are the
//!    `PlainLayout` length scalars CubeCL registers, one per view. M1 supports
//!    `u32` scalars only; any other scalar element type is rejected loudly.
//! 3. **Info buffer.** When the kernel carries any scalar, a trailing GLOBAL,
//!    read-only `u32` pointer is appended: the info buffer the CUDA server
//!    binds last (`cubecl-cuda/src/compute/server.rs`: buffers, then info).
//!
//! ## Info-buffer layout (findings sections 4/5)
//!
//! ```text
//! info[0 .. scalars_count)            = PlainLayout scalar lengths (one per view)
//! info[scalars_count .. +buffers_len) = static per-buffer lengths
//! ```
//!
//! `Metadata::BufferLength { var: GlobalBuffer(i) }` therefore reads
//! `info[scalars_count + i]`. The dynamic (shape/stride) section is empty
//! because no buffer carries extended metadata in this kernel.
//!
//! `convert_function_type` in mir-lower lowers each `MirPtrType` to a single
//! `.param .u64` and each scalar integer to a by-value `.param`, so the PTX
//! entry's parameters are the buffer base addresses, the scalar lengths, then
//! the info-buffer base address, in this order.

use cubecl_core::ir::{ElemType, FloatKind, StorageType, UIntKind};
use cubecl_runtime::kernel::KernelDefinition;
use dialect_mir::types::MirPtrType;
use pliron::{
    builtin::types::{FP32Type, IntegerType, Signedness},
    context::Context,
    r#type::TypeHandle,
};

/// Map a CubeCL element type to its dialect-mir scalar type.
///
/// M1 supports `f32` only; any other element type fails loudly.
pub(crate) fn mir_elem_type(ctx: &mut Context, elem: ElemType) -> Result<TypeHandle, String> {
    match elem {
        ElemType::Float(FloatKind::F32) => Ok(FP32Type::get(ctx).into()),
        other => Err(format!("unsupported element type: {other:?} (M1 supports f32 only)")),
    }
}

/// The dialect-mir type for a kernel `u32` scalar and for metadata length
/// values: a 32-bit signless integer. This matches the absolute-position index
/// type, so the bounds-check comparison needs no cast.
pub(crate) fn u32_mir_type(ctx: &mut Context) -> TypeHandle {
    IntegerType::get(ctx, 32, Signedness::Signless).into()
}

/// Build a GLOBAL, mutable pointer to `elem` (the kernel buffer pointer type).
pub(crate) fn global_ptr_type(ctx: &mut Context, elem: ElemType) -> Result<TypeHandle, String> {
    let pointee = mir_elem_type(ctx, elem)?;
    Ok(MirPtrType::get_global(ctx, pointee, true).into())
}

/// The info-buffer pointer type: a GLOBAL, read-only `u32` pointer.
pub(crate) fn info_ptr_type(ctx: &mut Context) -> TypeHandle {
    let pointee = u32_mir_type(ctx);
    MirPtrType::get_global(ctx, pointee, false).into()
}

/// Total number of scalar parameter slots. Each `ScalarKernelArg { count }`
/// expands to `count` slots.
pub(crate) fn scalars_count(kernel: &KernelDefinition) -> usize {
    kernel.scalars.iter().map(|s| s.count).sum()
}

/// `INFO_ALIGN` from `cubecl-core/src/codegen/info.rs`: the info buffer is a
/// `Vec<u64>`, so every section is padded to an 8-byte boundary.
const INFO_ALIGN: usize = 8;

/// The u32 index at which the static metadata (per-buffer lengths) begins inside
/// the info buffer, i.e. the size of the scalar section in u32 slots.
///
/// `cubecl`'s `Info::new` lays the info buffer out as `[scalars | static_meta |
/// dynamic_meta]` over a `Vec<u64>`. Each scalar group of `count` elements of a
/// given type occupies `(ty.size() * count)` bytes rounded up to `INFO_ALIGN`,
/// so 3 `u32` scalars take `(4*3).next_multiple_of(8) = 16` bytes = 4 u32 slots
/// (3 values + 1 padding). The buffer-length for buffer `i` then lives at
/// `meta_base_index + i` (`Metadata::buffer_len_index(i) = i`). Mirroring this
/// exactly is what gives bit-identical metadata reads versus nvrtc.
pub(crate) fn meta_base_index(kernel: &KernelDefinition) -> usize {
    let scalar_bytes: usize = kernel
        .scalars
        .iter()
        .map(|s| (s.ty.size() * s.count).next_multiple_of(INFO_ALIGN))
        .sum();
    scalar_bytes / 4
}

/// Whether the kernel carries an info buffer (a trailing GLOBAL `u32` pointer).
/// Present whenever the kernel has scalar params, matching CubeCL's binding
/// order. The M1 spine has no scalars and thus no info buffer.
pub(crate) fn has_info_buffer(kernel: &KernelDefinition) -> bool {
    scalars_count(kernel) > 0
}

/// Kernel-parameter types: buffer pointers, then `u32` scalar values, then the
/// info-buffer pointer (when scalars are present). Rejects tensor maps and
/// non-`u32` scalars, which are out of M1 scope.
pub(crate) fn kernel_param_types(
    ctx: &mut Context,
    kernel: &KernelDefinition,
) -> Result<Vec<TypeHandle>, String> {
    if !kernel.tensor_maps.is_empty() {
        return Err(format!(
            "unsupported kernel: {} tensor-map argument(s) (M1 spine has none)",
            kernel.tensor_maps.len()
        ));
    }
    for scalar in &kernel.scalars {
        match scalar.ty {
            StorageType::Scalar(ElemType::UInt(UIntKind::U32)) => {}
            other => {
                return Err(format!(
                    "unsupported scalar type: {other:?} (M1 supports u32 scalars only)"
                ));
            }
        }
    }

    // Kernel parameters are the buffer pointers followed by the trailing info
    // buffer pointer. Scalars are NOT separate parameters: cubecl folds the
    // `PlainLayout` length scalars into the info buffer (its leading
    // `scalars_count` slots), and the server binds only `buffers ++ info`
    // (no per-scalar binding; `grid_constants` is disabled so info is a plain
    // GLOBAL `u32` device pointer). The walk reads each `GlobalScalar(id)` from
    // `info[id]` accordingly.
    let s_count = scalars_count(kernel);
    let mut params = Vec::with_capacity(kernel.buffers.len() + 1);
    for buffer in &kernel.buffers {
        params.push(global_ptr_type(ctx, buffer.ty.elem_type())?);
    }
    if s_count > 0 {
        params.push(info_ptr_type(ctx));
    }
    Ok(params)
}

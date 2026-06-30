//! Kernel ABI: mapping CubeCL `KernelDefinition` buffers/scalars to the
//! dialect-mir function-parameter types the walk builds the entry block from.
//!
//! ## Mapping (M1 spine)
//!
//! Each global buffer becomes exactly one `MirPtrType` in the GLOBAL address
//! space, in declaration order (`kernel.buffers[0]` -> param 0, ...). The
//! pointee is the buffer's element type (`f32` only for M1). Every buffer
//! pointer is emitted `is_mutable = true`: `KernelArg` carries no per-buffer
//! visibility in this CubeCL revision, so we take the writable superset, which
//! still permits loads from read-only inputs.
//!
//! `convert_function_type` in mir-lower lowers each `MirPtrType{addrspace}` to
//! a single `.param .u64`, so an N-buffer kernel yields an N-pointer PTX entry
//! whose parameters are the buffer base addresses in declaration order.
//!
//! Scalars are appended after buffers in the full ABI; the M1 spine carries
//! none, so a non-empty scalar list is rejected loudly rather than silently
//! dropped.

use cubecl_core::ir::{ElemType, FloatKind};
use cubecl_runtime::kernel::KernelDefinition;
use dialect_mir::types::MirPtrType;
use pliron::{
    builtin::types::FP32Type,
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

/// Build a GLOBAL, mutable pointer to `elem` (the kernel buffer pointer type).
pub(crate) fn global_ptr_type(ctx: &mut Context, elem: ElemType) -> Result<TypeHandle, String> {
    let pointee = mir_elem_type(ctx, elem)?;
    Ok(MirPtrType::get_global(ctx, pointee, true).into())
}

/// Kernel-parameter pointer types, one GLOBAL `MirPtrType` per buffer in
/// declaration order. Rejects kernels carrying scalars or tensor maps, which
/// are out of M1 spine scope.
pub(crate) fn kernel_param_types(
    ctx: &mut Context,
    kernel: &KernelDefinition,
) -> Result<Vec<TypeHandle>, String> {
    if !kernel.scalars.is_empty() {
        return Err(format!(
            "unsupported kernel: {} scalar argument(s) (M1 spine has none)",
            kernel.scalars.len()
        ));
    }
    if !kernel.tensor_maps.is_empty() {
        return Err(format!(
            "unsupported kernel: {} tensor-map argument(s) (M1 spine has none)",
            kernel.tensor_maps.len()
        ));
    }

    let mut params = Vec::with_capacity(kernel.buffers.len());
    for buffer in &kernel.buffers {
        params.push(global_ptr_type(ctx, buffer.ty.elem_type())?);
    }
    Ok(params)
}

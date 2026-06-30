//! Walk a CubeCL `KernelDefinition` for the elementwise-add spine into a
//! dialect-mir module, mirroring the hand-built recipe proved in cuda-oxide's
//! `spine_kernel_ptx` test but driven from CubeCL IR.
//!
//! The spine is `out[pos] = a[pos] + b[pos]` at `line_size = 1`, `f32`:
//!
//! - `VariableKind::GlobalBuffer(id)` -> a GLOBAL `f32` pointer kernel param
//!   (one block argument per buffer, in declaration order).
//! - `Builtin::AbsolutePos` (or its `X` component) -> `ctaid.x * ntid.x + tid.x`
//!   via the nvvm special-register reads plus `mir.mul` / `mir.add`.
//! - `Memory::Index` -> `mir.ptr_offset` (a GEP producing a `f32*`).
//! - `Memory::Load` -> `mir.load`.
//! - `Memory::Store` -> `mir.store` (operands `[dest_ptr, value]`).
//! - `Arithmetic::Add` -> `mir.add`.
//! - end of body -> `mir.return`.
//!
//! Every `Operation` variant outside this set returns `Err` (fail loud); there
//! is no panic and no silent skip.

use std::any::TypeId;
use std::collections::HashMap;

use cubecl_core::ir::{
    Arithmetic, Builtin, ElemType, Memory, Operation, Variable, VariableKind,
};
use cubecl_runtime::kernel::KernelDefinition;
use dialect_mir::ops::{
    MirAddOp, MirFuncOp, MirLoadOp, MirMulOp, MirPtrOffsetOp, MirReturnOp, MirStoreOp,
};
use dialect_nvvm::ops::{ReadPtxSregCtaidXOp, ReadPtxSregNtidXOp, ReadPtxSregTidXOp};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{StringAttr, TypeAttr},
        op_interfaces::SymbolOpInterface,
        ops::ModuleOp,
        types::{FunctionType, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::{Op, OpObj},
    operation::Operation as MirOperation,
    r#type::TypeHandle,
    value::Value,
};

use super::abi;

/// Concrete-op-info pair returned by every op's `get_concrete_op_info`.
type OpInfo = (fn(Ptr<MirOperation>) -> OpObj, TypeId);

/// Append a 0-region op to `block` and return its single result (if any).
fn emit(
    ctx: &mut Context,
    block: Ptr<BasicBlock>,
    info: OpInfo,
    results: Vec<TypeHandle>,
    operands: Vec<Value>,
) -> Option<Value> {
    let op = MirOperation::new(ctx, info, results.clone(), operands, vec![], 0);
    let res = if results.is_empty() {
        None
    } else {
        Some(op.deref(ctx).get_result(0))
    };
    op.insert_at_back(block, ctx);
    res
}

/// The kernel entry symbol name: the kernel's configured name, or `kernel`.
pub fn entry_name(kernel: &KernelDefinition) -> String {
    let name = kernel.options.kernel_name.trim();
    if name.is_empty() {
        "kernel".to_string()
    } else {
        name.to_string()
    }
}

/// Per-kernel lowering state.
struct Lowering {
    entry: Ptr<BasicBlock>,
    /// SSA map from CubeCL variable kind to the pliron value holding it.
    values: HashMap<VariableKind, Value>,
    /// Cached `ctaid.x * ntid.x + tid.x` absolute-position index.
    abs_pos: Option<Value>,
    i32_ty: TypeHandle,
}

impl Lowering {
    /// Resolve a CubeCL operand to its pliron value, materialising the
    /// absolute-position index on first use.
    fn resolve(&mut self, ctx: &mut Context, var: &Variable) -> Result<Value, String> {
        match var.kind {
            VariableKind::Builtin(Builtin::AbsolutePos)
            | VariableKind::Builtin(Builtin::AbsolutePosX) => Ok(self.absolute_pos(ctx)),
            kind => self
                .values
                .get(&kind)
                .copied()
                .ok_or_else(|| format!("unresolved variable: {kind:?}")),
        }
    }

    /// `i = ctaid.x * ntid.x + tid.x`, computed once and cached.
    fn absolute_pos(&mut self, ctx: &mut Context) -> Value {
        if let Some(v) = self.abs_pos {
            return v;
        }
        let i32_ty = self.i32_ty;
        let tid = emit(
            ctx,
            self.entry,
            ReadPtxSregTidXOp::get_concrete_op_info(),
            vec![i32_ty],
            vec![],
        )
        .expect("tid read has a result");
        let ctaid = emit(
            ctx,
            self.entry,
            ReadPtxSregCtaidXOp::get_concrete_op_info(),
            vec![i32_ty],
            vec![],
        )
        .expect("ctaid read has a result");
        let ntid = emit(
            ctx,
            self.entry,
            ReadPtxSregNtidXOp::get_concrete_op_info(),
            vec![i32_ty],
            vec![],
        )
        .expect("ntid read has a result");
        let block_base = emit(
            ctx,
            self.entry,
            MirMulOp::get_concrete_op_info(),
            vec![i32_ty],
            vec![ctaid, ntid],
        )
        .expect("mul has a result");
        let idx = emit(
            ctx,
            self.entry,
            MirAddOp::get_concrete_op_info(),
            vec![i32_ty],
            vec![block_base, tid],
        )
        .expect("add has a result");
        self.abs_pos = Some(idx);
        idx
    }

    /// Bind the output variable of an instruction to a freshly produced value.
    fn bind(&mut self, out: Option<Variable>, value: Value) -> Result<(), String> {
        let out = out.ok_or("value-producing operation has no output variable")?;
        self.values.insert(out.kind, value);
        Ok(())
    }
}

/// Lower a spine `KernelDefinition` into a dialect-mir `ModuleOp`, returning the
/// module operation pointer ready for `compile_module_to_ptx`.
pub fn lower_kernel(
    ctx: &mut Context,
    kernel: &KernelDefinition,
) -> Result<Ptr<MirOperation>, String> {
    let param_types = abi::kernel_param_types(ctx, kernel)?;
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();

    // Module shell.
    let module = ModuleOp::new(ctx, "kernel_module".try_into().unwrap());
    let module_op = module.get_operation();
    let module_region = module_op.deref(ctx).get_region(0);
    let module_block = {
        let existing = {
            let region = module_region.deref(ctx);
            region.iter(ctx).next()
        };
        existing.unwrap_or_else(|| {
            let block = BasicBlock::new(ctx, None, vec![]);
            block.insert_at_back(module_region, ctx);
            block
        })
    };

    // Function `(buffers...) -> void`.
    let func_type = FunctionType::get(ctx, param_types.clone(), vec![]);
    let func = {
        let op = MirOperation::new(ctx, MirFuncOp::get_concrete_op_info(), vec![], vec![], vec![], 1);
        let func = MirFuncOp::new(ctx, op, TypeAttr::new(func_type.into()));
        let name = entry_name(kernel);
        func.set_symbol_name(
            ctx,
            name.as_str()
                .try_into()
                .map_err(|_| format!("invalid kernel symbol name: {name:?}"))?,
        );
        func
    };

    // Mark as a PTX `.visible .entry`: presence of a `gpu_kernel` StringAttr is
    // all `is_kernel_func` inspects.
    {
        let key: pliron::identifier::Identifier = "gpu_kernel".try_into().unwrap();
        func.get_operation()
            .deref_mut(ctx)
            .attributes
            .set(key, StringAttr::new("true".to_string()));
    }

    // Entry block whose arguments are the buffer pointers.
    let entry = BasicBlock::new(ctx, None, param_types);
    let func_region = func.get_operation().deref(ctx).get_region(0);
    entry.insert_at_front(func_region, ctx);

    let mut state = Lowering {
        entry,
        values: HashMap::new(),
        abs_pos: None,
        i32_ty,
    };
    for (i, buffer) in kernel.buffers.iter().enumerate() {
        let arg = entry.deref(ctx).get_argument(i);
        state
            .values
            .insert(VariableKind::GlobalBuffer(buffer.id), arg);
    }

    // Walk the body.
    let instructions = kernel.body.instructions.borrow();
    for inst in instructions.iter() {
        lower_instruction(ctx, &mut state, inst)?;
    }
    drop(instructions);

    // Terminate with `return`.
    emit(
        ctx,
        entry,
        MirReturnOp::get_concrete_op_info(),
        vec![],
        vec![],
    );

    func.get_operation().insert_at_back(module_block, ctx);
    Ok(module_op)
}

/// Lower a single CubeCL instruction, extending the SSA map.
fn lower_instruction(
    ctx: &mut Context,
    state: &mut Lowering,
    inst: &cubecl_core::ir::Instruction,
) -> Result<(), String> {
    match &inst.operation {
        // `&list[index]` -> ptr_offset (GEP) producing a GLOBAL f32 pointer.
        Operation::Memory(Memory::Index(op)) => {
            let base = state.resolve(ctx, &op.list)?;
            let index = state.resolve(ctx, &op.index)?;
            let ptr_ty = abi::global_ptr_type(ctx, op.list.ty.elem_type())?;
            let res = emit(
                ctx,
                state.entry,
                MirPtrOffsetOp::get_concrete_op_info(),
                vec![ptr_ty],
                vec![base, index],
            )
            .expect("ptr_offset has a result");
            state.bind(inst.out, res)
        }

        // `load(ptr)` -> mir.load.
        Operation::Memory(Memory::Load(ptr)) => {
            let ptr_val = state.resolve(ctx, ptr)?;
            let elem_ty = elem_mir_type(ctx, inst, ptr)?;
            let res = emit(
                ctx,
                state.entry,
                MirLoadOp::get_concrete_op_info(),
                vec![elem_ty],
                vec![ptr_val],
            )
            .expect("load has a result");
            state.bind(inst.out, res)
        }

        // `store(ptr, value)` -> mir.store, operands [dest_ptr, value].
        Operation::Memory(Memory::Store(op)) => {
            let ptr_val = state.resolve(ctx, &op.ptr)?;
            let value = state.resolve(ctx, &op.value)?;
            emit(
                ctx,
                state.entry,
                MirStoreOp::get_concrete_op_info(),
                vec![],
                vec![ptr_val, value],
            );
            Ok(())
        }

        // `lhs + rhs` -> mir.add.
        Operation::Arithmetic(Arithmetic::Add(op)) => {
            let lhs = state.resolve(ctx, &op.lhs)?;
            let rhs = state.resolve(ctx, &op.rhs)?;
            let out = inst.out.ok_or("Arithmetic::Add has no output variable")?;
            let ty = abi::mir_elem_type(ctx, out.ty.elem_type())?;
            let res = emit(
                ctx,
                state.entry,
                MirAddOp::get_concrete_op_info(),
                vec![ty],
                vec![lhs, rhs],
            )
            .expect("add has a result");
            state.bind(inst.out, res)
        }

        // Loud failure on every other arithmetic op.
        Operation::Arithmetic(other) => {
            Err(format!("unsupported operation: Arithmetic::{other:?}"))
        }

        // Loud failure on every other memory op.
        Operation::Memory(other) => Err(format!("unsupported operation: Memory::{other:?}")),

        // Loud failure on every other operation kind.
        other => Err(format!("unsupported operation: {other:?}")),
    }
}

/// The dialect-mir element type a load produces: from its output variable when
/// present, else the pointed-to buffer's element type.
fn elem_mir_type(
    ctx: &mut Context,
    inst: &cubecl_core::ir::Instruction,
    ptr: &Variable,
) -> Result<TypeHandle, String> {
    let elem: ElemType = inst
        .out
        .map(|o| o.ty.elem_type())
        .unwrap_or_else(|| ptr.ty.elem_type());
    abi::mir_elem_type(ctx, elem)
}

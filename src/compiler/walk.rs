//! Walk a CubeCL `KernelDefinition` for the full `Tensor::add` op surface into a
//! dialect-mir module, driven from CubeCL IR.
//!
//! Beyond the elementwise-add spine (`out[pos] = a[pos] + b[pos]`, `f32`,
//! `line_size = 1`) this also covers the metadata reads, slice-aggregate
//! bookkeeping, and the bounds-check control flow that a real `Tensor::add`
//! `KernelDefinition` carries:
//!
//! - `VariableKind::GlobalBuffer(id)` -> a GLOBAL `f32` pointer kernel param.
//! - `VariableKind::GlobalScalar(id)` -> a by-value `i32` kernel param (the
//!   `PlainLayout` length scalars).
//! - the trailing info buffer -> a GLOBAL `u32` pointer kernel param.
//! - `Builtin::AbsolutePos` -> `ctaid.x * ntid.x + tid.x`.
//! - `Operation::ConstructAggregate` / `ExtractAggregateField` -> folded at
//!   lowering time: an aggregate is a virtual tuple of already-lowered values,
//!   so no MIR op or allocation is emitted (the slice offset is constant 0).
//! - `Metadata::BufferLength { var: GlobalBuffer(i) }` -> a load from the info
//!   buffer at `scalars_count + i` (see `abi.rs`).
//! - `Comparison::Lower` -> `mir.lt` (a direct `i1` `a < b`).
//! - `Operator::And` / `Operator::Not` on `i1` -> `mir.bitand` / `mir.not`.
//! - `Branch::If { cond, { Branch::Return } }` -> the `terminate!()` bounds
//!   guard: a `mir.cond_br` splitting into a return block and a continue block.
//! - `Memory::Index` -> `mir.ptr_offset`; `Memory::Load` -> `mir.load`;
//!   `Memory::Store` -> `mir.store`; `Arithmetic::Add` -> `mir.add`.
//! - end of body -> `mir.return`.
//!
//! Every `Operation` (and nested `Arithmetic` / `Comparison` / `Operator` /
//! `Branch` / `Memory` / `Metadata`) variant outside this set returns `Err`
//! (fail loud); there is no panic and no silent skip.

use std::any::TypeId;
use std::collections::HashMap;
use std::num::NonZero;

use cubecl_core::ir::{
    Arithmetic, Branch, Builtin, Comparison, ElemType, Memory, Metadata, Operation, Operator,
    Variable, VariableKind,
};
use cubecl_runtime::kernel::KernelDefinition;
use dialect_mir::ops::{
    MirAddOp, MirBitAndOp, MirConstantOp, MirCondBranchOp, MirFuncOp, MirLoadOp, MirLtOp, MirMulOp,
    MirNotOp, MirPtrOffsetOp, MirReturnOp, MirStoreOp,
};
use dialect_nvvm::ops::{ReadPtxSregCtaidXOp, ReadPtxSregNtidXOp, ReadPtxSregTidXOp};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, StringAttr, TypeAttr},
        op_interfaces::{OperandSegmentInterface, SymbolOpInterface},
        ops::ModuleOp,
        types::{FunctionType, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    linked_list::ContainsLinkedList,
    op::{Op, OpObj},
    operation::Operation as MirOperation,
    region::Region,
    r#type::TypeHandle,
    utils::apint::APInt,
    value::Value,
};

use super::abi;

/// Concrete-op-info pair returned by every op's `get_concrete_op_info`.
type OpInfo = (fn(Ptr<MirOperation>) -> OpObj, TypeId);

/// A lowered CubeCL value: either a single pliron value or a virtual aggregate
/// (slice tuple). Aggregates are folded entirely at lowering time, so they hold
/// the original element variables and are dereferenced on field extraction.
#[derive(Clone)]
enum Lowered {
    Val(Value),
    Aggregate(Vec<Variable>),
}

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
    /// The block ops are currently appended to. Starts as the entry block and
    /// moves to the continue block once a bounds-check branch splits the body.
    current: Ptr<BasicBlock>,
    /// The function body region, where extra basic blocks are inserted.
    func_region: Ptr<Region>,
    /// SSA map from CubeCL variable kind to its lowered value (or aggregate).
    values: HashMap<VariableKind, Lowered>,
    /// Cached `ctaid.x * ntid.x + tid.x` absolute-position index.
    abs_pos: Option<Value>,
    /// The info-buffer pointer param (present iff the kernel carries scalars).
    info_ptr: Option<Value>,
    /// Number of scalar slots, the base offset of buffer lengths in the info
    /// buffer.
    scalars_count: usize,
    i32_ty: TypeHandle,
    i1_ty: TypeHandle,
}

impl Lowering {
    /// Resolve a CubeCL operand to its single pliron value, materialising the
    /// absolute-position index on first use. Aggregates cannot be used as a
    /// scalar value and fail loud.
    fn resolve(&mut self, ctx: &mut Context, var: &Variable) -> Result<Value, String> {
        match var.kind {
            VariableKind::Builtin(Builtin::AbsolutePos)
            | VariableKind::Builtin(Builtin::AbsolutePosX) => Ok(self.absolute_pos(ctx)),
            kind => match self.values.get(&kind) {
                Some(Lowered::Val(v)) => Ok(*v),
                Some(Lowered::Aggregate(_)) => {
                    Err(format!("cannot use aggregate {kind:?} as a scalar value"))
                }
                None => Err(format!("unresolved variable: {kind:?}")),
            },
        }
    }

    /// `i = ctaid.x * ntid.x + tid.x`, computed once into the current block and
    /// cached.
    fn absolute_pos(&mut self, ctx: &mut Context) -> Value {
        if let Some(v) = self.abs_pos {
            return v;
        }
        let i32_ty = self.i32_ty;
        let block = self.current;
        let tid = emit(
            ctx,
            block,
            ReadPtxSregTidXOp::get_concrete_op_info(),
            vec![i32_ty],
            vec![],
        )
        .expect("tid read has a result");
        let ctaid = emit(
            ctx,
            block,
            ReadPtxSregCtaidXOp::get_concrete_op_info(),
            vec![i32_ty],
            vec![],
        )
        .expect("ctaid read has a result");
        let ntid = emit(
            ctx,
            block,
            ReadPtxSregNtidXOp::get_concrete_op_info(),
            vec![i32_ty],
            vec![],
        )
        .expect("ntid read has a result");
        let block_base = emit(
            ctx,
            block,
            MirMulOp::get_concrete_op_info(),
            vec![i32_ty],
            vec![ctaid, ntid],
        )
        .expect("mul has a result");
        let idx = emit(
            ctx,
            block,
            MirAddOp::get_concrete_op_info(),
            vec![i32_ty],
            vec![block_base, tid],
        )
        .expect("add has a result");
        self.abs_pos = Some(idx);
        idx
    }

    /// Emit an `i32` constant into the current block.
    fn iconst(&mut self, ctx: &mut Context, value: i64) -> Value {
        let i32_ty = self.i32_ty;
        let op = MirOperation::new(
            ctx,
            MirConstantOp::get_concrete_op_info(),
            vec![i32_ty],
            vec![],
            vec![],
            0,
        );
        let int_ty = IntegerType::get(ctx, 32, Signedness::Signless);
        let attr = IntegerAttr::new(int_ty, APInt::from_i64(value, NonZero::new(32).unwrap()));
        MirConstantOp::new(op).set_attr_value(ctx, attr);
        op.insert_at_back(self.current, ctx);
        op.deref(ctx).get_result(0)
    }

    /// Bind an instruction output variable to a single value.
    fn bind_val(&mut self, out: Option<Variable>, value: Value) -> Result<(), String> {
        let out = out.ok_or("value-producing operation has no output variable")?;
        self.values.insert(out.kind, Lowered::Val(value));
        Ok(())
    }
}

/// Lower a `KernelDefinition` into a dialect-mir `ModuleOp`, returning the
/// module operation pointer ready for `compile_module_to_ptx`.
pub fn lower_kernel(
    ctx: &mut Context,
    kernel: &KernelDefinition,
) -> Result<Ptr<MirOperation>, String> {
    let param_types = abi::kernel_param_types(ctx, kernel)?;
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let i1_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();

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

    // Function `(params...) -> void`.
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

    // Entry block whose arguments are the kernel params (buffers, scalars, info).
    let entry = BasicBlock::new(ctx, None, param_types);
    let func_region = func.get_operation().deref(ctx).get_region(0);
    entry.insert_at_front(func_region, ctx);

    let scalars_count = abi::scalars_count(kernel);
    let mut state = Lowering {
        current: entry,
        func_region,
        values: HashMap::new(),
        abs_pos: None,
        info_ptr: None,
        scalars_count,
        i32_ty,
        i1_ty,
    };
    let n_buf = kernel.buffers.len();
    for (i, buffer) in kernel.buffers.iter().enumerate() {
        let arg = entry.deref(ctx).get_argument(i);
        state
            .values
            .insert(VariableKind::GlobalBuffer(buffer.id), Lowered::Val(arg));
    }
    for i in 0..scalars_count {
        let arg = entry.deref(ctx).get_argument(n_buf + i);
        state
            .values
            .insert(VariableKind::GlobalScalar(i as u32), Lowered::Val(arg));
    }
    if abi::has_info_buffer(kernel) {
        state.info_ptr = Some(entry.deref(ctx).get_argument(n_buf + scalars_count));
    }

    // Walk the body.
    let instructions = kernel.body.instructions.borrow();
    for inst in instructions.iter() {
        lower_instruction(ctx, &mut state, inst)?;
    }
    drop(instructions);

    // Terminate the final (continue) block with `return`.
    emit(
        ctx,
        state.current,
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
        // Slice aggregate construction folds to a virtual tuple: no op emitted.
        Operation::ConstructAggregate(elements) => {
            let out = inst
                .out
                .ok_or("ConstructAggregate has no output variable")?;
            state
                .values
                .insert(out.kind, Lowered::Aggregate(elements.clone()));
            Ok(())
        }

        // Field extraction reads back the (already-lowered) aggregate element.
        Operation::ExtractAggregateField(op) => {
            let elem_var = match state.values.get(&op.aggregate.kind) {
                Some(Lowered::Aggregate(elements)) => elements.get(op.field).copied().ok_or_else(
                    || format!("aggregate field {} out of range", op.field),
                )?,
                Some(Lowered::Val(_)) => {
                    return Err(format!(
                        "ExtractAggregateField on non-aggregate {:?}",
                        op.aggregate.kind
                    ));
                }
                None => {
                    return Err(format!("unresolved aggregate: {:?}", op.aggregate.kind));
                }
            };
            let lowered = state
                .values
                .get(&elem_var.kind)
                .cloned()
                .ok_or_else(|| format!("unresolved aggregate element: {:?}", elem_var.kind))?;
            let out = inst
                .out
                .ok_or("ExtractAggregateField has no output variable")?;
            state.values.insert(out.kind, lowered);
            Ok(())
        }

        // Buffer length -> load from the info buffer at `scalars_count + i`.
        Operation::Metadata(Metadata::BufferLength { var }) => {
            let info = state.info_ptr.ok_or(
                "BufferLength requires an info-buffer param, but the kernel carries no scalars",
            )?;
            let id = match var.kind {
                VariableKind::GlobalBuffer(id) => id as usize,
                other => {
                    return Err(format!("unsupported BufferLength target: {other:?}"));
                }
            };
            let index = state.scalars_count + id;
            let idx_val = state.iconst(ctx, index as i64);
            let info_ptr_ty = abi::info_ptr_type(ctx);
            let elem_ptr = emit(
                ctx,
                state.current,
                MirPtrOffsetOp::get_concrete_op_info(),
                vec![info_ptr_ty],
                vec![info, idx_val],
            )
            .expect("ptr_offset has a result");
            let len = emit(
                ctx,
                state.current,
                MirLoadOp::get_concrete_op_info(),
                vec![state.i32_ty],
                vec![elem_ptr],
            )
            .expect("load has a result");
            state.bind_val(inst.out, len)
        }

        // `&list[index]` -> ptr_offset (GEP) producing a GLOBAL f32 pointer.
        Operation::Memory(Memory::Index(op)) => {
            let base = state.resolve(ctx, &op.list)?;
            let index = state.resolve(ctx, &op.index)?;
            let ptr_ty = abi::global_ptr_type(ctx, op.list.ty.elem_type())?;
            let res = emit(
                ctx,
                state.current,
                MirPtrOffsetOp::get_concrete_op_info(),
                vec![ptr_ty],
                vec![base, index],
            )
            .expect("ptr_offset has a result");
            state.bind_val(inst.out, res)
        }

        // `load(ptr)` -> mir.load.
        Operation::Memory(Memory::Load(ptr)) => {
            let ptr_val = state.resolve(ctx, ptr)?;
            let elem_ty = elem_mir_type(ctx, inst, ptr)?;
            let res = emit(
                ctx,
                state.current,
                MirLoadOp::get_concrete_op_info(),
                vec![elem_ty],
                vec![ptr_val],
            )
            .expect("load has a result");
            state.bind_val(inst.out, res)
        }

        // `store(ptr, value)` -> mir.store, operands [dest_ptr, value].
        Operation::Memory(Memory::Store(op)) => {
            let ptr_val = state.resolve(ctx, &op.ptr)?;
            let value = state.resolve(ctx, &op.value)?;
            emit(
                ctx,
                state.current,
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
                state.current,
                MirAddOp::get_concrete_op_info(),
                vec![ty],
                vec![lhs, rhs],
            )
            .expect("add has a result");
            state.bind_val(inst.out, res)
        }

        // `lhs < rhs` -> mir.lt (a direct i1 result).
        Operation::Comparison(Comparison::Lower(op)) => {
            let lhs = state.resolve(ctx, &op.lhs)?;
            let rhs = state.resolve(ctx, &op.rhs)?;
            let i1_ty = state.i1_ty;
            let res = emit(
                ctx,
                state.current,
                MirLtOp::get_concrete_op_info(),
                vec![i1_ty],
                vec![lhs, rhs],
            )
            .expect("lt has a result");
            state.bind_val(inst.out, res)
        }

        // `lhs & rhs` on i1 predicates -> mir.bitand.
        Operation::Operator(Operator::And(op)) => {
            let lhs = state.resolve(ctx, &op.lhs)?;
            let rhs = state.resolve(ctx, &op.rhs)?;
            let i1_ty = state.i1_ty;
            let res = emit(
                ctx,
                state.current,
                MirBitAndOp::get_concrete_op_info(),
                vec![i1_ty],
                vec![lhs, rhs],
            )
            .expect("bitand has a result");
            state.bind_val(inst.out, res)
        }

        // `!val` on an i1 predicate -> mir.not.
        Operation::Operator(Operator::Not(op)) => {
            let input = state.resolve(ctx, &op.input)?;
            let i1_ty = state.i1_ty;
            let res = emit(
                ctx,
                state.current,
                MirNotOp::get_concrete_op_info(),
                vec![i1_ty],
                vec![input],
            )
            .expect("not has a result");
            state.bind_val(inst.out, res)
        }

        // `if cond { return }` (the terminate!() bounds guard) -> cond_br
        // splitting into a return block and a continue block.
        Operation::Branch(Branch::If(if_box)) => {
            // Only the `terminate!()` shape is supported: the body must be a
            // single `Branch::Return`. Anything else fails loud.
            let body_is_return = {
                let body = if_box.scope.instructions.borrow();
                body.len() == 1
                    && matches!(body[0].operation, Operation::Branch(Branch::Return))
            };
            if !body_is_return {
                return Err(
                    "unsupported Branch::If body: only terminate!() (a single Branch::Return) is supported"
                        .to_string(),
                );
            }

            let cond = state.resolve(ctx, &if_box.cond)?;

            // True side: return block.
            let ret_block = BasicBlock::new(ctx, None, vec![]);
            ret_block.insert_at_back(state.func_region, ctx);
            emit(
                ctx,
                ret_block,
                MirReturnOp::get_concrete_op_info(),
                vec![],
                vec![],
            );

            // False side: continue block, where the rest of the body lands.
            let cont_block = BasicBlock::new(ctx, None, vec![]);
            cont_block.insert_at_back(state.func_region, ctx);

            let (flat, sizes) =
                MirCondBranchOp::compute_segment_sizes(vec![vec![cond], vec![], vec![]]);
            let br = MirOperation::new(
                ctx,
                MirCondBranchOp::get_concrete_op_info(),
                vec![],
                flat,
                vec![ret_block, cont_block],
                0,
            );
            MirCondBranchOp::new(br).set_operand_segment_sizes(ctx, sizes);
            br.insert_at_back(state.current, ctx);

            state.current = cont_block;
            Ok(())
        }

        // Loud failure on every other branch op.
        Operation::Branch(other) => Err(format!("unsupported operation: Branch::{other:?}")),

        // Loud failure on every other comparison op.
        Operation::Comparison(other) => {
            Err(format!("unsupported operation: Comparison::{other:?}"))
        }

        // Loud failure on every other operator.
        Operation::Operator(other) => Err(format!("unsupported operation: Operator::{other:?}")),

        // Loud failure on every other metadata op.
        Operation::Metadata(other) => Err(format!("unsupported operation: Metadata::{other:?}")),

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

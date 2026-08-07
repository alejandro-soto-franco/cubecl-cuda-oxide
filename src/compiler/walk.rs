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
//!   buffer at `meta_base + i` (meta_base = padded scalar-section size in u32 slots; see `abi.rs`).
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
    Arithmetic, Branch, Builtin, Comparison, ConstantValue, ElemType, Memory, Metadata, Operation,
    Operator, Variable, VariableKind,
};
use cubecl_runtime::kernel::KernelDefinition;
use dialect_mir::ops::{
    MirAddOp, MirBitAndOp, MirConstantOp, MirCondBranchOp, MirFuncOp, MirLoadOp, MirLtOp, MirMulOp,
    MirNotOp, MirPtrOffsetOp, MirReturnOp, MirStoreOp,
};
use dialect_nvvm::ops::{
    ReadPtxSregCtaidXOp, ReadPtxSregCtaidYOp, ReadPtxSregCtaidZOp, ReadPtxSregNctaidXOp,
    ReadPtxSregNctaidYOp, ReadPtxSregNtidXOp, ReadPtxSregNtidYOp, ReadPtxSregNtidZOp,
    ReadPtxSregTidXOp, ReadPtxSregTidYOp, ReadPtxSregTidZOp,
};
use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::{IntegerAttr, TypeAttr},
        op_interfaces::{OperandSegmentInterface, SymbolOpInterface},
        types::{FunctionType, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    op::{Op, OpObj},
    operation::Operation as MirOperation,
    region::Region,
    r#type::TypeHandle,
    utils::apint::APInt,
    value::Value,
};

use super::abi;

/// The pair every op's `get_concrete_op_info` returns.
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
    /// u32 index inside the info buffer at which the per-buffer length metadata
    /// begins (the scalar section size in u32 slots, padded to 8 bytes). See
    /// `abi::meta_base_index`.
    meta_base: usize,
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
            // Inline constant operands (e.g. the `PlainLayout` slice offset
            // `const 0`) are materialised on demand as a MIR constant. M1 only
            // needs integer constants (slice offsets / indices, lowered into the
            // `i32` index domain). Float/bool constant operands do not occur in
            // `Tensor::add`, so they fail loud rather than guess a type.
            VariableKind::Constant(c) => match c {
                ConstantValue::Int(v) => Ok(self.iconst(ctx, v)),
                ConstantValue::UInt(v) => Ok(self.iconst(ctx, v as i64)),
                other => Err(format!(
                    "unsupported constant operand {other:?}: only integer constants are lowered in M1"
                )),
            },
            // `PlainLayout` length scalars are not kernel params; read them from
            // the info buffer at `info[id]` (the leading scalars section).
            VariableKind::GlobalScalar(id) => self.info_load(ctx, id as usize),
            kind => match self.values.get(&kind) {
                Some(Lowered::Val(v)) => Ok(*v),
                Some(Lowered::Aggregate(_)) => {
                    Err(format!("cannot use aggregate {kind:?} as a scalar value"))
                }
                None => Err(format!("unresolved variable: {kind:?}")),
            },
        }
    }

    /// The flattened global thread index `ABSOLUTE_POS` (cubecl's `idxGlobal`),
    /// computed once into the current block and cached.
    ///
    /// burn launches elementwise kernels with multi-dimensional cube (block) and
    /// grid shapes (`CubeDim::new` returns a 2D `plane_size x plane_count`
    /// block), so the X-only `ctaid.x * ntid.x + tid.x` is wrong whenever the
    /// block or grid extends past X. This reproduces cubecl-cpp's exact formula
    /// (`cubecl-cpp/src/shared/kernel.rs`):
    ///
    /// ```text
    /// ap{d}    = ctaid.{d} * ntid.{d} + tid.{d}      // per-dimension
    /// idxGlobal = ap_z * (nctaid.x*ntid.x*nctaid.y*ntid.y)
    ///           + ap_y * (nctaid.x*ntid.x)
    ///           + ap_x
    /// ```
    ///
    /// All arithmetic is in the `i32` index domain, matching the f32 element size
    /// `mul.wide` indexing that follows. Integer arithmetic, so the grouping is
    /// exact.
    fn absolute_pos(&mut self, ctx: &mut Context) -> Value {
        if let Some(v) = self.abs_pos {
            return v;
        }
        // Per-dimension absolute positions: ctaid.d * ntid.d + tid.d.
        let tid_x = self.sreg(ctx, ReadPtxSregTidXOp::get_concrete_op_info());
        let tid_y = self.sreg(ctx, ReadPtxSregTidYOp::get_concrete_op_info());
        let tid_z = self.sreg(ctx, ReadPtxSregTidZOp::get_concrete_op_info());
        let ctaid_x = self.sreg(ctx, ReadPtxSregCtaidXOp::get_concrete_op_info());
        let ctaid_y = self.sreg(ctx, ReadPtxSregCtaidYOp::get_concrete_op_info());
        let ctaid_z = self.sreg(ctx, ReadPtxSregCtaidZOp::get_concrete_op_info());
        let ntid_x = self.sreg(ctx, ReadPtxSregNtidXOp::get_concrete_op_info());
        let ntid_y = self.sreg(ctx, ReadPtxSregNtidYOp::get_concrete_op_info());
        let ntid_z = self.sreg(ctx, ReadPtxSregNtidZOp::get_concrete_op_info());
        let nctaid_x = self.sreg(ctx, ReadPtxSregNctaidXOp::get_concrete_op_info());
        let nctaid_y = self.sreg(ctx, ReadPtxSregNctaidYOp::get_concrete_op_info());

        let ap_x = {
            let m = self.muli(ctx, ctaid_x, ntid_x);
            self.addi(ctx, m, tid_x)
        };
        let ap_y = {
            let m = self.muli(ctx, ctaid_y, ntid_y);
            self.addi(ctx, m, tid_y)
        };
        let ap_z = {
            let m = self.muli(ctx, ctaid_z, ntid_z);
            self.addi(ctx, m, tid_z)
        };

        // stride_y = cube_count_x * cube_dim_x; stride_z = stride_y * cube_count_y * cube_dim_y.
        let stride_y = self.muli(ctx, nctaid_x, ntid_x);
        let stride_z = {
            let cc_y_cd_y = self.muli(ctx, nctaid_y, ntid_y);
            self.muli(ctx, stride_y, cc_y_cd_y)
        };

        let term_z = self.muli(ctx, ap_z, stride_z);
        let term_y = self.muli(ctx, ap_y, stride_y);
        let idx = {
            let zy = self.addi(ctx, term_z, term_y);
            self.addi(ctx, zy, ap_x)
        };

        self.abs_pos = Some(idx);
        idx
    }

    /// Read a 32-bit special register (tid/ctaid/ntid/nctaid component).
    fn sreg(&mut self, ctx: &mut Context, info: OpInfo) -> Value {
        emit(ctx, self.current, info, vec![self.i32_ty], vec![]).expect("sreg read has a result")
    }

    /// `a * b` in the i32 index domain.
    fn muli(&mut self, ctx: &mut Context, a: Value, b: Value) -> Value {
        emit(
            ctx,
            self.current,
            MirMulOp::get_concrete_op_info(),
            vec![self.i32_ty],
            vec![a, b],
        )
        .expect("mul has a result")
    }

    /// `a + b` in the i32 index domain.
    fn addi(&mut self, ctx: &mut Context, a: Value, b: Value) -> Value {
        emit(
            ctx,
            self.current,
            MirAddOp::get_concrete_op_info(),
            vec![self.i32_ty],
            vec![a, b],
        )
        .expect("add has a result")
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

    /// Load a 32-bit value from the info buffer at `index` (a u32-element
    /// offset, not bytes). The info buffer is the trailing GLOBAL `u32` pointer
    /// param laid out as `[scalars | static_meta | dynamic_meta]`: scalar
    /// `GlobalScalar(id)` is at `info[id]`, and buffer length `id` is at
    /// `info[meta_base + id]` (see `abi::meta_base_index`).
    fn info_load(&mut self, ctx: &mut Context, index: usize) -> Result<Value, String> {
        let info = self.info_ptr.ok_or(
            "info-buffer access requires an info-buffer param, but the kernel carries none",
        )?;
        let idx_val = self.iconst(ctx, index as i64);
        let info_ptr_ty = abi::info_ptr_type(ctx);
        let elem_ptr = emit(
            ctx,
            self.current,
            MirPtrOffsetOp::get_concrete_op_info(),
            vec![info_ptr_ty],
            vec![info, idx_val],
        )
        .expect("ptr_offset has a result");
        let val = emit(
            ctx,
            self.current,
            MirLoadOp::get_concrete_op_info(),
            vec![self.i32_ty],
            vec![elem_ptr],
        )
        .expect("load has a result");
        Ok(val)
    }

    /// Bind an instruction output variable to a single value.
    fn bind_val(&mut self, out: Option<Variable>, value: Value) -> Result<(), String> {
        let out = out.ok_or("value-producing operation has no output variable")?;
        self.values.insert(out.kind, Lowered::Val(value));
        Ok(())
    }
}

/// Lower a `KernelDefinition` into a `MirFuncOp`, appended to `module_block`.
///
/// The `gpu_kernel` marker is not set here: the caller marks the entry
/// through `CodegenModule::mark_kernel_entry` once `edit` returns.
pub fn lower_kernel(
    ctx: &mut Context,
    module_block: Ptr<BasicBlock>,
    kernel: &KernelDefinition,
) -> Result<(), String> {
    let param_types = abi::kernel_param_types(ctx, kernel)?;
    let i32_ty: TypeHandle = IntegerType::get(ctx, 32, Signedness::Signless).into();
    let i1_ty: TypeHandle = IntegerType::get(ctx, 1, Signedness::Signless).into();

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

    // Entry block whose arguments are the kernel params (buffers, scalars, info).
    let entry = BasicBlock::new(ctx, None, param_types);
    let func_region = func.get_operation().deref(ctx).get_region(0);
    entry.insert_at_front(func_region, ctx);

    let mut state = Lowering {
        current: entry,
        func_region,
        values: HashMap::new(),
        abs_pos: None,
        info_ptr: None,
        meta_base: abi::meta_base_index(kernel),
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
    // Scalars are not kernel params; they live in the info buffer and are read on
    // demand (see `GlobalScalar` in `resolve`). The info-buffer pointer is the
    // single trailing param, right after the buffers.
    if abi::has_info_buffer(kernel) {
        state.info_ptr = Some(entry.deref(ctx).get_argument(n_buf));
    }

    // Walk the body. Lowering errors are annotated with the offending
    // instruction index and operation so a fail-loud `Err` pinpoints the op.
    let instructions = kernel.body.instructions.borrow();
    for (i, inst) in instructions.iter().enumerate() {
        lower_instruction(ctx, &mut state, inst).map_err(|e| {
            format!("{e} (while lowering body instruction {i}: {:?})", inst.operation)
        })?;
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
    Ok(())
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
            // The element is usually a previously-lowered value or a nested
            // aggregate. It can also be an inline constant (the slice offset
            // `const 0`), which is never recorded in `values`; materialise it
            // through `resolve` in that case.
            let lowered = match state.values.get(&elem_var.kind) {
                Some(l) => l.clone(),
                None => Lowered::Val(state.resolve(ctx, &elem_var)?),
            };
            let out = inst
                .out
                .ok_or("ExtractAggregateField has no output variable")?;
            state.values.insert(out.kind, lowered);
            Ok(())
        }

        // Buffer length -> load from the info buffer at `scalars_count + id`.
        Operation::Metadata(Metadata::BufferLength { var }) => {
            let id = match var.kind {
                VariableKind::GlobalBuffer(id) => id as usize,
                other => {
                    return Err(format!("unsupported BufferLength target: {other:?}"));
                }
            };
            let len = state.info_load(ctx, state.meta_base + id)?;
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

        // `lhs + rhs` -> mir.add. Add appears in two domains in a real
        // `Tensor::add`: the f32 element sum (`a + b`) and the integer index
        // arithmetic `slice_offset + ABSOLUTE_POS` (u32, lowered into the i32
        // index domain that `ptr_offset`/`ABSOLUTE_POS` already use).
        Operation::Arithmetic(Arithmetic::Add(op)) => {
            let lhs = state.resolve(ctx, &op.lhs)?;
            let rhs = state.resolve(ctx, &op.rhs)?;
            let out = inst.out.ok_or("Arithmetic::Add has no output variable")?;
            let ty = arith_result_type(ctx, state, out.ty.elem_type())?;
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

/// Result type for an arithmetic op, selecting the right domain: `f32` element
/// arithmetic maps to `FP32`, integer/unsigned (index) arithmetic maps to the
/// `i32` index domain shared by `ABSOLUTE_POS` and `ptr_offset`. Other element
/// types are out of M1 scope and fail loud.
fn arith_result_type(
    ctx: &mut Context,
    state: &Lowering,
    elem: ElemType,
) -> Result<TypeHandle, String> {
    match elem {
        ElemType::Float(_) => abi::mir_elem_type(ctx, elem),
        ElemType::Int(_) | ElemType::UInt(_) => Ok(state.i32_ty),
        other => Err(format!(
            "unsupported arithmetic result type: {other:?} (M1 supports f32 and integer index math)"
        )),
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

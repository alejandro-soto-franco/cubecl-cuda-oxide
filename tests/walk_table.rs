//! Table-driven lowering test for the elementwise-add spine.
//!
//! A minimal `KernelDefinition` for `out[pos] = a[pos] + b[pos]` (line_size 1,
//! f32) is hand-built from CubeCL IR, lowered via `walk::lower_kernel`, compiled
//! to PTX, and validated: `.visible .entry` present, `.target sm_120`, and
//! `ptxas -arch=sm_120` accepts the assembly.

// `mir-importer` links the compiler's `rustc_driver` dylib; any binary that
// links it (this external test) must enable `rustc_private` and force the
// single dylib copy, else linking fails with "X only shows up once".
#![feature(rustc_private)]

extern crate rustc_driver;

use cubecl_core::ir::{
    Arithmetic, BinaryOperands, Branch, Builtin, Comparison, ElemType, FloatKind, If, IndexOperands,
    Instruction, Memory, Metadata, Operator, Scope, StorageType, StoreOperands, Type, UIntKind,
    UnaryOperands, Variable, VariableKind,
};
use cubecl_runtime::kernel::{KernelArg, KernelDefinition, KernelOptions, ScalarKernelArg};
use cubecl_runtime::server::CubeDim;

use cubecl_cuda_oxide::compiler::walk;

fn f32_type() -> Type {
    Type::new(StorageType::Scalar(ElemType::Float(FloatKind::F32)))
}

fn u32_type() -> Type {
    Type::new(StorageType::Scalar(ElemType::UInt(UIntKind::U32)))
}

/// Build the spine `out[pos] = a[pos] + b[pos]` kernel definition by hand.
///
/// Instruction stream (mirrors what an elementwise add lowers to):
///   ref_a = &a[pos]; a_i = load(ref_a)
///   ref_b = &b[pos]; b_i = load(ref_b)
///   sum   = a_i + b_i
///   ref_o = &out[pos]; store(ref_o, sum)
fn build_spine_add_kernel() -> KernelDefinition {
    let f32_ty = f32_type();

    let a = Variable::new(VariableKind::GlobalBuffer(0), f32_ty);
    let b = Variable::new(VariableKind::GlobalBuffer(1), f32_ty);
    let out = Variable::new(VariableKind::GlobalBuffer(2), f32_ty);
    let pos = Variable::new(VariableKind::Builtin(Builtin::AbsolutePos), u32_type());

    // Local SSA values (types beyond their element type are not inspected by
    // the walk; f32 scalars keep the stream simple).
    let ref_a = Variable::new(VariableKind::LocalConst { id: 0 }, f32_ty);
    let a_i = Variable::new(VariableKind::LocalConst { id: 1 }, f32_ty);
    let ref_b = Variable::new(VariableKind::LocalConst { id: 2 }, f32_ty);
    let b_i = Variable::new(VariableKind::LocalConst { id: 3 }, f32_ty);
    let sum = Variable::new(VariableKind::LocalConst { id: 4 }, f32_ty);
    let ref_o = Variable::new(VariableKind::LocalConst { id: 5 }, f32_ty);

    let index = |list: Variable| {
        Memory::Index(IndexOperands {
            list,
            index: pos,
            vector_size: 0,
            unroll_factor: 1,
            checked: false,
        })
    };

    let scope = Scope::root(false);
    {
        let mut ins = scope.instructions.borrow_mut();
        ins.push(Instruction::new(index(a), ref_a));
        ins.push(Instruction::new(Memory::Load(ref_a), a_i));
        ins.push(Instruction::new(index(b), ref_b));
        ins.push(Instruction::new(Memory::Load(ref_b), b_i));
        ins.push(Instruction::new(
            Arithmetic::Add(BinaryOperands { lhs: a_i, rhs: b_i }),
            sum,
        ));
        ins.push(Instruction::new(index(out), ref_o));
        ins.push(Instruction::no_out(Memory::Store(StoreOperands {
            ptr: ref_o,
            value: sum,
        })));
    }

    let buffer = |id| KernelArg {
        id,
        ty: f32_ty,
        has_extended_meta: false,
    };

    KernelDefinition {
        buffers: vec![buffer(0), buffer(1), buffer(2)],
        tensor_maps: vec![],
        scalars: vec![],
        cube_dim: CubeDim { x: 256, y: 1, z: 1 },
        body: scope,
        options: KernelOptions {
            kernel_name: "add_kernel".to_string(),
            ..Default::default()
        },
    }
}

/// Run `ptxas -arch=<arch>` over the PTX; return success and captured stderr.
fn ptxas_accepts(ptx: &[u8], arch: &str) -> (bool, String) {
    // Unique per invocation: tests run in parallel and must not share a path.
    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let uniq = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!("walk_table_{}_{}", std::process::id(), uniq));
    std::fs::create_dir_all(&dir).unwrap();
    let ptx_path = dir.join("k.ptx");
    let cubin_path = dir.join("k.cubin");
    std::fs::write(&ptx_path, ptx).unwrap();

    let ptxas = if std::path::Path::new("/usr/local/cuda/bin/ptxas").exists() {
        "/usr/local/cuda/bin/ptxas"
    } else {
        "ptxas"
    };
    let out = std::process::Command::new(ptxas)
        .arg(format!("-arch={arch}"))
        .arg("--compile-only")
        .arg(&ptx_path)
        .arg("-o")
        .arg(&cubin_path)
        .output()
        .expect("ptxas runs");

    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let _ = std::fs::remove_dir_all(&dir);
    (out.status.success(), stderr)
}

/// One f32 `GlobalBuffer`.
fn f32_buffer(id: u32) -> KernelArg {
    KernelArg {
        id,
        ty: f32_type(),
        has_extended_meta: false,
    }
}

/// A `ScalarKernelArg` of `count` U32 PlainLayout lengths.
fn u32_scalars(count: usize) -> ScalarKernelArg {
    ScalarKernelArg {
        ty: StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
        count,
    }
}

fn pos_var() -> Variable {
    Variable::new(VariableKind::Builtin(Builtin::AbsolutePos), u32_type())
}

fn local(id: u32) -> Variable {
    Variable::new(VariableKind::LocalConst { id }, f32_type())
}

fn index_at(list: Variable, index: Variable) -> Memory {
    Memory::Index(IndexOperands {
        list,
        index,
        vector_size: 0,
        unroll_factor: 1,
        checked: false,
    })
}

/// Build a `KernelDefinition` from a finished scope, with the given buffers and
/// scalar count.
fn kernel(name: &str, buffers: Vec<KernelArg>, scalars: usize, scope: Scope) -> KernelDefinition {
    let scalar_args = if scalars == 0 {
        vec![]
    } else {
        vec![u32_scalars(scalars)]
    };
    KernelDefinition {
        buffers,
        tensor_maps: vec![],
        scalars: scalar_args,
        cube_dim: CubeDim { x: 256, y: 1, z: 1 },
        body: scope,
        options: KernelOptions {
            kernel_name: name.to_string(),
            ..Default::default()
        },
    }
}

/// Lower a kernel and compile to PTX text, propagating lowering errors.
fn lower_to_ptx(kernel: &KernelDefinition) -> Result<(Vec<u8>, String), String> {
    let mut ctx = pliron::context::Context::new();
    mir_importer::translator::register_dialects(&mut ctx);
    let module_op = walk::lower_kernel(&mut ctx, kernel)?;
    let ptx = mir_importer::compile_module_to_ptx(
        &mut ctx,
        module_op,
        &mir_importer::PtxConfig::new("sm_120"),
    )
    .map_err(|e| format!("compile failed: {e:?}"))?;
    let text = String::from_utf8(ptx.clone()).map_err(|e| e.to_string())?;
    Ok((ptx, text))
}

/// `Metadata::BufferLength`: read the info buffer. The kernel reads `in[len]`
/// and writes it to `out[pos]` (a genuine cross-buffer copy, so the info-buffer
/// load survives dead-code elimination).
fn build_buffer_length_kernel() -> KernelDefinition {
    let in_buf = f32_buffer(0);
    let out_buf = f32_buffer(1);
    let in_var = Variable::new(VariableKind::GlobalBuffer(0), f32_type());
    let out_var = Variable::new(VariableKind::GlobalBuffer(1), f32_type());
    let len = Variable::new(VariableKind::LocalConst { id: 0 }, u32_type());
    let refp = local(1);
    let v = local(2);
    let refo = local(3);

    let scope = Scope::root(false);
    {
        let mut ins = scope.instructions.borrow_mut();
        ins.push(Instruction::new(
            Metadata::BufferLength { var: in_var },
            len,
        ));
        ins.push(Instruction::new(index_at(in_var, len), refp));
        ins.push(Instruction::new(Memory::Load(refp), v));
        ins.push(Instruction::new(index_at(out_var, pos_var()), refo));
        ins.push(Instruction::no_out(Memory::Store(StoreOperands {
            ptr: refo,
            value: v,
        })));
    }
    kernel("buffer_length_kernel", vec![in_buf, out_buf], 1, scope)
}

/// Bounds-check control flow: `Comparison::Lower` x2 -> `Operator::And` ->
/// `Operator::Not` -> `Branch::If { Branch::Return }`, then the guarded body.
fn build_bounds_check_kernel() -> KernelDefinition {
    let a = f32_buffer(0);
    let b = f32_buffer(1);
    let out = f32_buffer(2);
    let a_var = Variable::new(VariableKind::GlobalBuffer(0), f32_type());
    let b_var = Variable::new(VariableKind::GlobalBuffer(1), f32_type());
    let out_var = Variable::new(VariableKind::GlobalBuffer(2), f32_type());
    let scalar0 = Variable::new(VariableKind::GlobalScalar(0), u32_type());
    let bool_t = u32_type();

    let len = Variable::new(VariableKind::LocalConst { id: 0 }, u32_type());
    let c1 = Variable::new(VariableKind::LocalConst { id: 1 }, bool_t);
    let c2 = Variable::new(VariableKind::LocalConst { id: 2 }, bool_t);
    let in_bounds = Variable::new(VariableKind::LocalConst { id: 3 }, bool_t);
    let nib = Variable::new(VariableKind::LocalConst { id: 4 }, bool_t);
    let ref_a = local(5);
    let va = local(6);
    let ref_b = local(7);
    let vb = local(8);
    let sum = local(9);
    let ref_o = local(10);

    let then_scope = Scope::root(false);
    then_scope
        .instructions
        .borrow_mut()
        .push(Instruction::no_out(Branch::Return));

    let scope = Scope::root(false);
    {
        let mut ins = scope.instructions.borrow_mut();
        // pos < scalar_len (PlainLayout) and pos < buffer_len (slice length).
        ins.push(Instruction::new(
            Comparison::Lower(BinaryOperands {
                lhs: pos_var(),
                rhs: scalar0,
            }),
            c1,
        ));
        ins.push(Instruction::new(
            Metadata::BufferLength { var: out_var },
            len,
        ));
        ins.push(Instruction::new(
            Comparison::Lower(BinaryOperands {
                lhs: pos_var(),
                rhs: len,
            }),
            c2,
        ));
        ins.push(Instruction::new(
            Operator::And(BinaryOperands { lhs: c1, rhs: c2 }),
            in_bounds,
        ));
        ins.push(Instruction::new(
            Operator::Not(UnaryOperands { input: in_bounds }),
            nib,
        ));
        ins.push(Instruction::no_out(Branch::If(Box::new(If {
            cond: nib,
            scope: then_scope,
        }))));
        // Guarded body: out[pos] = a[pos] + b[pos].
        ins.push(Instruction::new(index_at(a_var, pos_var()), ref_a));
        ins.push(Instruction::new(Memory::Load(ref_a), va));
        ins.push(Instruction::new(index_at(b_var, pos_var()), ref_b));
        ins.push(Instruction::new(Memory::Load(ref_b), vb));
        ins.push(Instruction::new(
            Arithmetic::Add(BinaryOperands { lhs: va, rhs: vb }),
            sum,
        ));
        ins.push(Instruction::new(index_at(out_var, pos_var()), ref_o));
        ins.push(Instruction::no_out(Memory::Store(StoreOperands {
            ptr: ref_o,
            value: sum,
        })));
    }
    kernel("bounds_check_kernel", vec![a, b, out], 1, scope)
}

/// Slice-aggregate folding: `ConstructAggregate` / `ExtractAggregateField` must
/// fold to no allocation; the surviving index+load lowers like the spine.
fn build_aggregate_kernel() -> KernelDefinition {
    let a = f32_buffer(0);
    let b = f32_buffer(1);
    let out = f32_buffer(2);
    let a_var = Variable::new(VariableKind::GlobalBuffer(0), f32_type());
    let b_var = Variable::new(VariableKind::GlobalBuffer(1), f32_type());
    let out_var = Variable::new(VariableKind::GlobalBuffer(2), f32_type());

    let len = Variable::new(VariableKind::LocalConst { id: 0 }, u32_type());
    let slice = Variable::new(VariableKind::LocalConst { id: 1 }, f32_type());
    let buf = local(2);
    let ref_a = local(3);
    let va = local(4);
    let ref_b = local(5);
    let vb = local(6);
    let sum = local(7);
    let ref_o = local(8);

    let scope = Scope::root(false);
    {
        let mut ins = scope.instructions.borrow_mut();
        ins.push(Instruction::new(
            Metadata::BufferLength { var: a_var },
            len,
        ));
        // slice = (list = a, offset = pos [unused field], length = len)
        ins.push(Instruction::new(
            cubecl_core::ir::Operation::ConstructAggregate(vec![a_var, pos_var(), len]),
            slice,
        ));
        // buf = slice.field(0) -> the list pointer (folds to `a`, no alloc).
        ins.push(Instruction::new(
            cubecl_core::ir::Operation::ExtractAggregateField(
                cubecl_core::ir::AggregateExtractOperands {
                    aggregate: slice,
                    field: 0,
                },
            ),
            buf,
        ));
        // out[pos] = buf[pos] + b[pos].
        ins.push(Instruction::new(index_at(buf, pos_var()), ref_a));
        ins.push(Instruction::new(Memory::Load(ref_a), va));
        ins.push(Instruction::new(index_at(b_var, pos_var()), ref_b));
        ins.push(Instruction::new(Memory::Load(ref_b), vb));
        ins.push(Instruction::new(
            Arithmetic::Add(BinaryOperands { lhs: va, rhs: vb }),
            sum,
        ));
        ins.push(Instruction::new(index_at(out_var, pos_var()), ref_o));
        ins.push(Instruction::no_out(Memory::Store(StoreOperands {
            ptr: ref_o,
            value: sum,
        })));
    }
    kernel("aggregate_kernel", vec![a, b, out], 1, scope)
}

/// A genuinely unsupported op (`Arithmetic::Mul`) to prove fail-loud fires.
fn build_unsupported_kernel() -> KernelDefinition {
    let a = local(0);
    let out = local(1);
    let scope = Scope::root(false);
    scope.instructions.borrow_mut().push(Instruction::new(
        Arithmetic::Mul(BinaryOperands { lhs: a, rhs: a }),
        out,
    ));
    kernel("unsupported_kernel", vec![f32_buffer(0)], 0, scope)
}

#[test]
fn buffer_length_lowers_with_info_load() {
    let kernel = build_buffer_length_kernel();
    let (ptx, text) = lower_to_ptx(&kernel).expect("buffer-length kernel lowers");
    let (ok, stderr) = ptxas_accepts(&ptx, "sm_120");
    assert!(ok, "ptxas rejected:\n{stderr}\n\nPTX:\n{text}");
    assert!(
        text.contains("ld.global"),
        "info-buffer load present:\n{text}"
    );
}

#[test]
fn bounds_check_lowers_with_conditional_branch() {
    let kernel = build_bounds_check_kernel();
    let (ptx, text) = lower_to_ptx(&kernel).expect("bounds-check kernel lowers");
    let (ok, stderr) = ptxas_accepts(&ptx, "sm_120");
    assert!(ok, "ptxas rejected:\n{stderr}\n\nPTX:\n{text}");
    assert!(
        text.contains("bra"),
        "conditional branch present:\n{text}"
    );
}

#[test]
fn aggregate_folds_with_no_allocation() {
    let kernel = build_aggregate_kernel();
    let (ptx, text) = lower_to_ptx(&kernel).expect("aggregate kernel lowers");
    let (ok, stderr) = ptxas_accepts(&ptx, "sm_120");
    assert!(ok, "ptxas rejected:\n{stderr}\n\nPTX:\n{text}");
    assert!(
        !text.contains(".local") && !text.contains("alloca"),
        "no aggregate allocation in PTX:\n{text}"
    );
    assert!(
        text.contains("ld.global"),
        "folded index+load present:\n{text}"
    );
}

#[test]
fn unsupported_operation_fails_loud() {
    let kernel = build_unsupported_kernel();
    let err = lower_to_ptx(&kernel).expect_err("unsupported op must fail loud");
    assert!(
        err.contains("unsupported operation"),
        "fail-loud message expected, got: {err}"
    );
}

#[test]
fn spine_add_lowers_and_validates() {
    let kernel = build_spine_add_kernel();

    let mut ctx = pliron::context::Context::new();
    mir_importer::translator::register_dialects(&mut ctx);

    let module_op = walk::lower_kernel(&mut ctx, &kernel).expect("spine lowers");

    let ptx = mir_importer::compile_module_to_ptx(
        &mut ctx,
        module_op,
        &mir_importer::PtxConfig::new("sm_120"),
    )
    .expect("module compiles to PTX");
    let text = String::from_utf8(ptx.clone()).expect("PTX is utf-8");

    assert!(
        text.contains(".visible .entry"),
        "kernel entry present:\n{text}"
    );
    assert!(
        text.contains(".target sm_120"),
        "PTX targets sm_120:\n{text}"
    );

    // Structural: the GLOBAL store and a fused-multiply-add index appear.
    assert!(text.contains("st.global"), "global store present:\n{text}");

    let (ok, stderr) = ptxas_accepts(&ptx, "sm_120");
    assert!(ok, "ptxas rejected PTX:\nstderr:\n{stderr}\n\nPTX:\n{text}");
}

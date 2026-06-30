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
    Arithmetic, BinaryOperands, Builtin, ElemType, FloatKind, IndexOperands, Instruction, Memory,
    Scope, StorageType, StoreOperands, Type, UIntKind, Variable, VariableKind,
};
use cubecl_runtime::kernel::{KernelArg, KernelDefinition, KernelOptions};
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
    let dir = std::env::temp_dir().join(format!("walk_table_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ptx_path = dir.join("spine.ptx");
    let cubin_path = dir.join("spine.cubin");
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

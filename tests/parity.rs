//! Numerical parity: prove `CudaOxide`'s elementwise `add` matches the stock
//! nvrtc backend (`burn_cuda::Cuda`) bit for bit, against a checked-in reference.
//!
//! Both backends are plain `CubeBackend<R>` (no fusion/autotune), so the only
//! difference exercised is the PTX producer: cuda-oxide's LLVM-NVPTX pipeline
//! versus nvrtc. The op surface is the same `kernel_binop<f32, Const<1>, AddOp>`.
//!
//! ## fp-contract
//!
//! Plain elementwise `add` lowers to a single `add.f32`; there is no
//! `fmul`+`fadd` pair for the backend to contract into an `fma`. fp-contract
//! (`-fp-contract=fast`) therefore cannot change the result, so cuda-oxide and
//! nvrtc are bit-exact here and the test asserts exact equality (no tolerance).
//! A kernel that does contain a multiply-add needs contraction disabled or a
//! `1e-6` tolerance before it can be compared this way.
//!
use burn_backend::{Backend, TensorData, read_sync};
use cubecl_cuda_oxide::CudaOxide;

const N: usize = 4096;
const FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/add_reference_f32.json"
);

/// Deterministic, seeded pseudo-random f32 inputs (xorshift64*; fixed seeds so
/// the fixture is reproducible without an RNG crate dependency).
fn seeded_inputs(n: usize) -> (Vec<f32>, Vec<f32>) {
    let make = |mut state: u64, n: usize| -> Vec<f32> {
        (0..n)
            .map(|_| {
                state ^= state >> 12;
                state ^= state << 25;
                state ^= state >> 27;
                let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
                // Map the top 24 bits to a float in [-128, 128) with a fractional part.
                let u = (bits >> 40) as f32 / (1u32 << 24) as f32; // [0, 1)
                (u - 0.5) * 256.0
            })
            .collect()
    };
    (make(0x1234_5678_9ABC_DEF0, n), make(0x0FED_CBA9_8765_4321, n))
}

/// Run `a + b` through backend `B`'s float op path and read the result back.
fn run_add<B: Backend>(a: &[f32], b: &[f32]) -> Vec<f32>
where
    B::Device: Default,
{
    let device = B::Device::default();
    let n = a.len();
    let ta = B::float_from_data(TensorData::new(a.to_vec(), [n]), &device);
    let tb = B::float_from_data(TensorData::new(b.to_vec(), [n]), &device);
    let tc = B::float_add(ta, tb);
    read_sync(B::float_into_data(tc))
        .expect("read back failed")
        .to_vec::<f32>()
        .expect("to_vec::<f32> failed")
}

/// The acceptance gate: CudaOxide's add matches the checked-in nvrtc reference,
/// bit for bit. Runs live on the GPU but needs no live nvrtc compile (the
/// reference is committed), so CI without an nvrtc toolchain still validates it.
#[test]
fn cuda_oxide_add_matches_nvrtc_reference() {
    // SAFETY: single-threaded test setup, before any backend work.
    unsafe {
        std::env::set_var("CUDA_OXIDE_TARGET", "sm_120");
    }

    let (a, b) = seeded_inputs(N);
    let out = run_add::<CudaOxide>(&a, &b);

    let reference: Vec<f32> = {
        let json = std::fs::read_to_string(FIXTURE)
            .unwrap_or_else(|e| panic!("missing fixture {FIXTURE}: {e}\nregenerate with: cargo test --test parity -- --ignored generate_nvrtc_reference"));
        serde_json::from_str(&json).expect("fixture is not a JSON array of f32")
    };

    assert_eq!(out.len(), reference.len(), "length mismatch");
    assert_eq!(out.len(), N, "expected {N} elements");

    let mut mismatches = 0usize;
    for (i, (o, r)) in out.iter().zip(reference.iter()).enumerate() {
        // Bit-exact: plain add has no fma to contract (see fp-contract note).
        if o.to_bits() != r.to_bits() {
            if mismatches < 8 {
                eprintln!("mismatch at {i}: oxide={o} ({:#x}) reference={r} ({:#x})", o.to_bits(), r.to_bits());
            }
            mismatches += 1;
        }
    }
    assert_eq!(mismatches, 0, "{mismatches}/{N} elements differ from the nvrtc reference");
}

/// Hardening test: exercises the runtime bounds-check early-return path in walk.rs.
///
/// N = 4097 is not a multiple of the cube dimension (256), so the last CUDA block
/// contains 255 out-of-bounds threads that must take the `Branch::If { Branch::Return }`
/// -> `cond_br` -> ret_block path. CPU `a[i] + b[i]` is the oracle (deterministic from
/// the same seed; validated bit-exact by `reference_fixture_is_correct`). Passing this
/// test proves the bounds-check guard fires correctly at runtime.
#[test]
fn cuda_oxide_add_bounds_check_oob_threads() {
    const N_OOB: usize = 4097; // not a multiple of cube_dim=256; last block has 255 oob threads
    // SAFETY: single-threaded test setup, before any backend work.
    unsafe {
        std::env::set_var("CUDA_OXIDE_TARGET", "sm_120");
    }

    let (a, b) = seeded_inputs(N_OOB);
    let out = run_add::<CudaOxide>(&a, &b);

    assert_eq!(out.len(), N_OOB, "length mismatch");

    let mut mismatches = 0usize;
    for i in 0..N_OOB {
        let cpu = a[i] + b[i];
        if out[i].to_bits() != cpu.to_bits() {
            if mismatches < 8 {
                eprintln!(
                    "mismatch at {i}: oxide={} ({:#x}) cpu={cpu} ({:#x})",
                    out[i],
                    out[i].to_bits(),
                    cpu.to_bits()
                );
            }
            mismatches += 1;
        }
    }
    assert_eq!(
        mismatches, 0,
        "{mismatches}/{N_OOB} elements differ from CPU reference (bounds-check path may be broken)"
    );
}

/// Cross-check the checked-in fixture against a CPU-computed sum, so a corrupt
/// or stale fixture cannot make the parity test pass vacuously.
#[test]
fn reference_fixture_is_correct() {
    let (a, b) = seeded_inputs(N);
    let reference: Vec<f32> =
        serde_json::from_str(&std::fs::read_to_string(FIXTURE).expect("fixture present"))
            .expect("fixture json");
    assert_eq!(reference.len(), N);
    for i in 0..N {
        assert_eq!(reference[i].to_bits(), (a[i] + b[i]).to_bits(), "fixture[{i}] != a+b on CPU");
    }
}

/// Regenerate the checked-in nvrtc reference. Ignored by default (needs a live
/// nvrtc toolchain). Run with:
/// `cargo test --test parity -- --ignored generate_nvrtc_reference`
#[test]
#[ignore = "regenerates the checked-in fixture via a live nvrtc run"]
fn generate_nvrtc_reference() {
    let (a, b) = seeded_inputs(N);
    let out = run_add::<burn_cuda::Cuda>(&a, &b);
    let json = serde_json::to_string(&out).expect("serialise");
    std::fs::write(FIXTURE, json).expect("write fixture");
    eprintln!("wrote {} elements to {FIXTURE}", out.len());
}

/// Live oxide-vs-nvrtc parity (both compiled fresh, same process). Ignored by
/// default because it needs a live nvrtc toolchain; the committed-fixture test
/// above is the CI gate.
#[test]
#[ignore = "needs a live nvrtc toolchain"]
fn live_parity_oxide_vs_nvrtc() {
    let (a, b) = seeded_inputs(N);
    let oxide = run_add::<CudaOxide>(&a, &b);
    let nvrtc = run_add::<burn_cuda::Cuda>(&a, &b);
    for i in 0..N {
        assert_eq!(oxide[i].to_bits(), nvrtc[i].to_bits(), "live mismatch at {i}");
    }
}

/// JIT-latency probe (cold first-compile vs warm cached) for both backends.
/// Ignored by default; run with:
/// `cargo test --test parity -- --ignored --nocapture jit_latency`
#[test]
#[ignore = "latency measurement, run on demand"]
fn jit_latency() {
    use std::time::Instant;
    // SAFETY: single-threaded test setup.
    unsafe {
        std::env::set_var("CUDA_OXIDE_TARGET", "sm_120");
    }
    let (a, b) = seeded_inputs(N);

    fn time_one<B: Backend>(a: &[f32], b: &[f32]) -> (f64, f64)
    where
        B::Device: Default,
    {
        let device = B::Device::default();
        let n = a.len();
        let mk = |s: &[f32]| B::float_from_data(TensorData::new(s.to_vec(), [n]), &device);

        // Cold: first add of this kernel triggers JIT compile + launch.
        let t0 = Instant::now();
        let c = B::float_add(mk(a), mk(b));
        let _ = read_sync(B::float_into_data(c));
        let cold = t0.elapsed().as_secs_f64() * 1e3;

        // Warm: identical kernel, served from the in-memory compiled-kernel cache.
        let t1 = Instant::now();
        let c = B::float_add(mk(a), mk(b));
        let _ = read_sync(B::float_into_data(c));
        let warm = t1.elapsed().as_secs_f64() * 1e3;

        (cold, warm)
    }

    let (ox_cold, ox_warm) = time_one::<CudaOxide>(&a, &b);
    let (nv_cold, nv_warm) = time_one::<burn_cuda::Cuda>(&a, &b);

    eprintln!("JIT latency (ms), N={N}, f32 add, sm_120");
    eprintln!("backend     cold      warm");
    eprintln!("cuda-oxide  {ox_cold:8.3}  {ox_warm:8.3}");
    eprintln!("nvrtc       {nv_cold:8.3}  {nv_warm:8.3}");
}

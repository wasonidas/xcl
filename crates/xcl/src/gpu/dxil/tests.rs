//! End-to-end checks of emitted DXIL: the SDK validator accepts it, DXC
//! disassembles it, and D3D12 runs it on the GPU with the recompiler's
//! kernel ABI (u0 `io`, u1 `res` root UAVs).

use super::container::Stage;
use super::module::{BinOp, CastOp, Pred};
use super::shader::Shader;
use super::validate;
use crate::gpu::recomp::runner;

/// The program of `tests/data/dxil/dxc_cs_raw_uav.hlsl`, emitted directly:
/// `a = io[lane]; f = asfloat(a) * 2 + 1; if (a > 5) f = -f;
///  res[lane] = asuint(f) + WaveReadLaneAt(a, 3)` — with a real branch and phi
/// rather than a select, to exercise control flow.
fn sample_kernel(wave: u32) -> Vec<u8> {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let io = s.raw_uav(0, 0);
    let res = s.raw_uav(1, 0);
    let lane = s.thread_id_in_group(0);
    let two = s.m.i32c(2);
    let off = s.m.bin(BinOp::Shl, lane, two);
    let a = s.raw_load(io, off, 1)[0];
    let f32 = s.m.f32();
    let i32 = s.m.i32();
    let fa = s.m.bitcast(a, f32);
    let c2 = s.m.f32c(2.0f32.to_bits());
    let c1 = s.m.f32c(1.0f32.to_bits());
    let mul = s.m.fbin(BinOp::Mul, fa, c2, 0);
    let f = s.m.fbin(BinOp::Add, mul, c1, 0);
    let five = s.m.i32c(5);
    let big = s.m.cmp(Pred::UGt, a, five);
    let three = s.m.i32c(3);
    let other = s.wave_read_lane_at(a, three);

    let entry = s.m.current_block();
    let neg_b = s.m.new_block();
    let join = s.m.new_block();
    s.m.cond_br(big, neg_b, join);
    s.m.set_block(neg_b);
    let nz = s.m.f32c(0x8000_0000);
    let neg = s.m.fbin(BinOp::Sub, nz, f, 0);
    s.m.br(join);
    s.m.set_block(join);
    let phi = s.m.phi(f32);
    s.m.add_incoming(phi, f, entry);
    s.m.add_incoming(phi, neg, neg_b);
    let bits = s.m.cast(CastOp::BitCast, phi, i32);
    let sum = s.m.bin(BinOp::Add, bits, other);
    s.raw_store(res, off, &[sum]);
    s.finish([64, 1, 1], Some(wave))
}

fn expected(io: &[u32], wave: usize) -> Vec<u32> {
    (0..64)
        .map(|l| {
            let a = io[l];
            let mut f = f32::from_bits(a) * 2.0 + 1.0;
            if a > 5 {
                f = -f;
            }
            f.to_bits().wrapping_add(io[l / wave * wave + 3])
        })
        .collect()
}

#[test]
fn emitted_kernel_passes_the_sdk_validator() {
    let c = sample_kernel(64);
    if let Err(e) = validate::validate(&c) {
        let dis = validate::disassemble(&c).unwrap_or_else(|d| d);
        panic!("validator rejected the emitted kernel:\n{e}\n--- disassembly ---\n{dis}");
    }
}

#[test]
fn emitted_kernel_disassembles_like_dxc_output() {
    let dis = validate::disassemble(&sample_kernel(64)).expect("disassembly");
    for needle in [
        "@dx.op.createHandleFromBinding(i32 217",
        "@dx.op.rawBufferLoad.i32(i32 139",
        "@dx.op.waveReadLaneAt.i32(i32 117",
        "phi float",
        "!{i32 64, i32 1, i32 1}",
    ] {
        assert!(dis.contains(needle), "missing `{needle}` in:\n{dis}");
    }
}

#[test]
fn emitted_kernel_runs_on_the_gpu() {
    let wave = match (runner::wave64_supported(), runner::wave32_supported()) {
        (Ok(true), _) => 64,
        (_, Ok(true)) => 32,
        (Err(e), _) => return runner::skip_test(&e),
        _ => return runner::skip_test("neither wave64 nor wave32 support"),
    };
    let c = sample_kernel(wave);
    validate::validate(&c).expect("valid DXIL");
    // io dwords 0..64 are the SGPR slots the runner uploads first.
    let mut sgpr = vec![0u32; 128];
    for (l, v) in sgpr.iter_mut().take(64).enumerate() {
        *v = match l % 4 {
            0 => l as u32,                     // small integers (≤ 5 stay positive)
            1 => (l as f32 * 0.5).to_bits(),   // floats, all > 5 as integers → negated
            2 => 3,
            _ => (-(l as f32)).to_bits(),
        };
    }
    let state = runner::KernelState { sgpr: sgpr.clone(), vgpr: vec![vec![0; 256]; 64], scc: 0 };
    let r = runner::run(&c, &state, &[]).expect("GPU run");
    assert_eq!(&r.raw[..64], &expected(&sgpr, wave as usize)[..]);
}

/// Blocks created out of dominance order (a block used as a branch target
/// before the block that dominates it exists), an unreachable block feeding a
/// phi, and `replace_all_uses` on a trivial phi: the writer lays blocks out in
/// reverse post-order, drops the unreachable one and its phi edge, and the
/// result validates and runs (res[lane] = (lane << 2) + 7 + io[lane]).
fn out_of_order_kernel() -> Vec<u8> {
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let io = s.raw_uav(0, 0);
    let res = s.raw_uav(1, 0);
    let lane = s.thread_id_in_group(0);
    let two = s.m.i32c(2);
    let off = s.m.bin(BinOp::Shl, lane, two);
    let entry = s.m.current_block();
    // created first, reached last
    let tail = s.m.new_block();
    let dead = s.m.new_block();
    let mid = s.m.new_block();
    s.m.br(mid);

    s.m.set_block(mid);
    let seven = s.m.i32c(7);
    let v = s.m.bin(BinOp::Add, off, seven);
    let x = s.raw_load(io, off, 1)[0];
    s.m.br(tail);

    s.m.set_block(dead);
    let junk = s.m.i32c(0xdead);
    s.m.br(tail);

    s.m.set_block(tail);
    let i32 = s.m.i32();
    let trivial = s.m.phi(i32); // only ever `v`: replaced below
    s.m.add_incoming(trivial, v, mid);
    let p = s.m.phi(i32);
    s.m.add_incoming(p, x, mid);
    s.m.add_incoming(p, junk, dead);
    let sum = s.m.bin(BinOp::Add, trivial, p);
    s.raw_store(res, off, &[sum]);
    s.m.replace_all_uses(trivial, v);
    let _ = entry;
    s.finish([64, 1, 1], None)
}

#[test]
fn blocks_are_written_in_reverse_post_order() {
    let c = out_of_order_kernel();
    if let Err(e) = validate::validate(&c) {
        let dis = validate::disassemble(&c).unwrap_or_else(|d| d);
        panic!("validator rejected the out-of-order kernel:\n{e}\n--- disassembly ---\n{dis}");
    }
    let dis = validate::disassemble(&c).expect("disassembly");
    assert!(!dis.contains("57005"), "the unreachable block's constant must be gone:\n{dis}");
    match runner::wave32_supported().or_else(|_| runner::wave64_supported()) {
        Ok(true) => {}
        Ok(false) => return runner::skip_test("no wave32 / wave64"),
        Err(e) => return runner::skip_test(&e),
    }
    let mut sgpr = vec![0u32; 128];
    for (l, v) in sgpr.iter_mut().take(64).enumerate() {
        *v = 1000 + l as u32 * 3;
    }
    let state = runner::KernelState { sgpr: sgpr.clone(), vgpr: vec![vec![0; 256]; 64], scc: 0 };
    let r = runner::run(&c, &state, &[]).expect("GPU run");
    let want: Vec<u32> = (0..64).map(|l| ((l as u32) << 2) + 7 + sgpr[l]).collect();
    assert_eq!(&r.raw[..64], &want[..]);
}

/// `replace_uses` makes a whole map of trivial phis in one pass and leaves the module as
/// `replace_all_uses` one at a time in order does, a chain included (`a → b`, then `b → v`,
/// as the SSA construction makes when `b` turns trivial after `a`).
#[test]
fn batched_replacements_match_one_at_a_time() {
    let build = |batched: bool| {
        let mut s = Shader::new(Stage::Compute, (6, 6));
        let io = s.raw_uav(0, 0);
        let res = s.raw_uav(1, 0);
        let lane = s.thread_id_in_group(0);
        let two = s.m.i32c(2);
        let off = s.m.bin(BinOp::Shl, lane, two);
        let next = s.m.new_block();
        let x = s.raw_load(io, off, 1)[0];
        s.m.br(next);
        let entry = s.m.entry_block();
        s.m.set_block(next);
        let i32 = s.m.i32();
        let b = s.m.phi(i32);
        s.m.add_incoming(b, x, entry);
        let a = s.m.phi(i32);
        s.m.add_incoming(a, b, entry);
        let sum = s.m.bin(BinOp::Add, a, b);
        s.raw_store(res, off, &[sum]);
        if batched {
            s.m.replace_uses(&std::collections::HashMap::from([(a, b), (b, x)]));
        } else {
            s.m.replace_all_uses(a, b);
            s.m.replace_all_uses(b, x);
        }
        s.finish([64, 1, 1], None)
    };
    let c = build(true);
    assert_eq!(c, build(false));
    if let Err(e) = validate::validate(&c) {
        panic!("validator rejected the kernel:\n{e}");
    }
}

fn sfi0_flags(dxil: &[u8]) -> u64 {
    let parts = crate::gpu::shader::container_parts(dxil).expect("valid DXBC container");
    let part = parts.iter().find(|part| part.fourcc == "SFI0").expect("SFI0 chunk");
    u64::from_le_bytes(dxil[part.offset..part.offset + part.size].try_into().expect("64-bit SFI0 flags"))
}

fn expected_msad_u8(a: u32, b: u32, c: u32) -> u32 {
    let mut sum = 0u32;
    for k in [0, 8, 16, 24] {
        let ba = (a >> k) & 0xFF;
        let bb = (b >> k) & 0xFF;
        if bb != 0 {
            sum += ba.abs_diff(bb);
        }
    }
    sum.wrapping_add(c)
}

#[test]
fn v_msad_u8_emits_dxil_msad_and_flags() {
    use super::kernel::{translate_full_as, KernelOptions};
    use super::ops::Tertiary;
    use crate::gpu::rdna2::encode;
    let enc = |n: &str, e: &str, f: &[(&str, u64)]| encode(n, e, f, None).unwrap();
    let v = |r: u64| 256 + r;
    let live = [
        enc("V_MSAD_U8", "ENC_VOP3", &[("VDST", 3), ("SRC0", v(0)), ("SRC1", v(1)), ("SRC2", v(2))]),
        enc("S_ENDPGM", "ENC_SOPP", &[]),
    ]
    .concat();
    let o = KernelOptions { wave64: true, split: true, mem_windows: 1, sampler_base: 0 };
    for compact in [false, true] {
        let k = translate_full_as(&live, o, compact).unwrap_or_else(|e| panic!("live compact={compact}: {e}"));
        if let Err(e) = validate::validate(&k.dxil) {
            let dis = validate::disassemble(&k.dxil).unwrap_or_else(|d| d);
            panic!("validator rejected live V_MSAD_U8 (compact={compact}):\n{e}\n--- disassembly ---\n{dis}");
        }
        assert_ne!(sfi0_flags(&k.dxil) & 0x40, 0, "compact={compact}: missing D3D_SHADER_REQUIRES_11_1_SHADER_EXTENSIONS in SFI0");
        let dis = validate::disassemble(&k.dxil).expect("disassembly");
        assert!(dis.contains("@dx.op.tertiary.i32(i32 50,"), "compact={compact}: missing Msad call:\n{dis}");
        assert!(dis.contains(", i32 0)  ; Msad("), "compact={compact}: Msad accum operand must be 0:\n{dis}");
        assert!(!dis.contains("@dx.op.binary.i32(i32 39,") && !dis.contains("@dx.op.binary.i32(i32 40,"),
            "compact={compact}: expanded UMax/UMin byte arithmetic should not be emitted:\n{dis}");
    }
    // A dead `Msad` call is removed by `Module::optimize` and must not leave a stale 11.1 flag.
    let mut s = Shader::new(Stage::Compute, (6, 6));
    let io = s.raw_uav(0, 0);
    let res = s.raw_uav(1, 0);
    let lane = s.thread_id_in_group(0);
    let two = s.m.i32c(2);
    let off = s.m.bin(BinOp::Shl, lane, two);
    let a = s.raw_load(io, off, 1)[0];
    let z = s.m.i32c(0);
    let _dead = s.tertiary(Tertiary::Msad, a, a, z);
    s.raw_store(res, off, &[a]);
    let dead_dxil = s.finish([64, 1, 1], None);
    if let Err(e) = validate::validate(&dead_dxil) {
        let dis = validate::disassemble(&dead_dxil).unwrap_or_else(|d| d);
        panic!("validator rejected dead Msad shader:\n{e}\n--- disassembly ---\n{dis}");
    }
    assert_eq!(sfi0_flags(&dead_dxil) & 0x40, 0, "dead Msad must not set 11.1 shader extensions flag");
    let dis_d = validate::disassemble(&dead_dxil).expect("disassembly");
    assert!(!dis_d.contains("@dx.op.tertiary.i32(i32 50,"), "dead Msad should be DCE'd:\n{dis_d}");
}

#[test]
fn gpu_v_msad_u8_matches_expected_in_plain_and_compact() {
    use super::kernel::{translate_full_as, KernelOptions};
    use crate::gpu::rdna2::encode;
    use crate::gpu::recomp::KERNEL_LANES;
    match (runner::adapter(), runner::wave64_programs_supported(), crate::gpu::recomp::dxc::location()) {
        (Ok(_), Ok(true), Ok(_)) => {}
        (a, _, d) => return runner::skip_test(if a.is_err() { "no GPU" } else if d.is_err() { "no DXC" } else { "no wave64" }),
    }
    let enc = |n: &str, e: &str, f: &[(&str, u64)]| encode(n, e, f, None).unwrap();
    let v = |r: u64| 256 + r;
    let code = [
        enc("V_MSAD_U8", "ENC_VOP3", &[("VDST", 3), ("SRC0", v(0)), ("SRC1", v(1)), ("SRC2", v(2))]),
        enc("V_MSAD_U8", "ENC_VOP3", &[("VDST", 4), ("SRC0", v(0)), ("SRC1", v(0)), ("SRC2", v(2))]),
        enc("V_MSAD_U8", "ENC_VOP3", &[("VDST", 5), ("SRC0", v(0)), ("SRC1", v(0)), ("SRC2", v(0))]),
        enc("V_MOV_B32", "ENC_VOP1", &[("VDST", 6), ("SRC0", v(0))]),
        enc("V_MSAD_U8", "ENC_VOP3", &[("VDST", 6), ("SRC0", v(6)), ("SRC1", v(1)), ("SRC2", v(2))]),
        enc("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", v(1))]),
        enc("V_MSAD_U8", "ENC_VOP3", &[("VDST", 7), ("SRC0", v(0)), ("SRC1", v(7)), ("SRC2", v(2))]),
        enc("V_MOV_B32", "ENC_VOP1", &[("VDST", 8), ("SRC0", v(2))]),
        enc("V_MSAD_U8", "ENC_VOP3", &[("VDST", 8), ("SRC0", v(0)), ("SRC1", v(1)), ("SRC2", v(8))]),
        enc("S_ENDPGM", "ENC_SOPP", &[]),
    ]
    .concat();
    let mut st = runner::KernelState::default();
    let fixed: &[(u32, u32, u32)] = &[
        (0xFF80_4020, 0x0010_0020, 100),                  // masked bytes in src1
        (0x0010_0020, 0xFF80_4020, 100),                  // zero bytes in src0 (unmasked)
        (0xFFFF_FFFF, 0x0000_0000, 0x1234_5678),          // all-zero mask
        (0x0000_0000, 0xFFFF_FFFF, 0),                    // maximum sum (1020)
        (0xFE80_7F01, 0x017F_80FE, 7),                    // high bytes (>= 0x80)
        (0x0000_0000, 0xFFFF_FFFF, 0u32.wrapping_sub(1020)), // max sum wraps exactly to 0
        (0x0000_0000, 0xFFFF_FFFF, 0xFFFF_FFFF),          // max sum wraps past u32::MAX
        (0xFFFF_FFFF, 0x0001_0001, 0xFFFF_FF00),          // partial mask + wrapping accum
        (0x80FF_017F, 0x80FF_017F, 0xFFFF_FFFE),          // tied src0 == src1 + high bytes
        (0x0000_0000, 0x0000_0000, 0x0000_0000),          // all zero
    ];
    for lane in 0..KERNEL_LANES {
        let (a, b, c) = if lane < fixed.len() {
            fixed[lane]
        } else {
            let l = lane as u32;
            let a = l.wrapping_mul(0x0103_070F) ^ 0x8000_0080;
            let b = if l % 3 == 0 { l.wrapping_mul(0x0011_0033) } else { l.wrapping_mul(0x0509_1121) ^ 0xFF00_8000 };
            let c = if l % 2 == 0 { 0xFFFF_FD00u32.wrapping_add(l * 17) } else { l * 13 };
            (a, b, c)
        };
        st.vgpr[lane][0] = a;
        st.vgpr[lane][1] = b;
        st.vgpr[lane][2] = c;
    }
    let o = KernelOptions { wave64: true, split: runner::split_wave64(), mem_windows: 1, sampler_base: 0 };
    for compact in [false, true] {
        let k = translate_full_as(&code, o, compact).unwrap_or_else(|e| panic!("compact={compact}: {e}"));
        validate::validate(&k.dxil).unwrap_or_else(|e| panic!("validate compact={compact}: {e}"));
        let out = runner::run_with_statics(&k.dxil, &st, &[], &[], &[], &k.static_samplers).expect("GPU run");
        for lane in 0..KERNEL_LANES {
            let (a, b, c) = (st.vgpr[lane][0], st.vgpr[lane][1], st.vgpr[lane][2]);
            let want_abc = expected_msad_u8(a, b, c);
            let want_aac = expected_msad_u8(a, a, c);
            let want_aaa = expected_msad_u8(a, a, a);
            assert_eq!(out.vgpr[lane][3], want_abc, "compact={compact} lane={lane} v3 (distinct)");
            assert_eq!(out.vgpr[lane][4], want_aac, "compact={compact} lane={lane} v4 (tied src0==src1)");
            assert_eq!(out.vgpr[lane][5], want_aaa, "compact={compact} lane={lane} v5 (tied all)");
            assert_eq!(out.vgpr[lane][6], want_abc, "compact={compact} lane={lane} v6 (vdst==src0)");
            assert_eq!(out.vgpr[lane][7], want_abc, "compact={compact} lane={lane} v7 (vdst==src1)");
            assert_eq!(out.vgpr[lane][8], want_abc, "compact={compact} lane={lane} v8 (vdst==src2)");
        }
    }
}

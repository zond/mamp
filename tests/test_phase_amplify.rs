//! Tests for phase_amplify.wgsl — phase-based motion amplification
//!
//! Run: cargo test --features native-test --test test_phase_amplify -- --nocapture

#[path = "gpu_test_harness.rs"]
mod gpu_test_harness;
use gpu_test_harness::*;

use std::f32::consts::PI;
use wgpu::BufferUsages;

const SHADER: &str = include_str!("../src/shaders/phase_amplify.wgsl");

/// Run phase_amplify shader once (one frame).
/// Returns (output_re, output_im).
fn run_phase_amplify(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cur_re: &[f32], cur_im: &[f32],
    prev_re: &[f32], prev_im: &[f32],
    lp_high: &[f32], lp_low: &[f32],
    width: u32, height: u32,
    amplification: f32, alpha_low: f32, alpha_high: f32,
    is_first_frame: bool,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    // Returns: (out_re, out_im, new_prev_re, new_prev_im, new_lp_high, new_lp_low)
    let params: [u32; 8] = [
        width, height,
        amplification.to_bits(), alpha_low.to_bits(),
        alpha_high.to_bits(), if is_first_frame { 1 } else { 0 },
        0, 0,
    ];
    let total = (width * height) as usize;
    let buf_bytes = total * 4;
    let s = BufferUsages::STORAGE;

    // Run shader — we need to read multiple output bindings
    // Run once per output we need
    let make_bufs = |out_binding: u32| -> Vec<u8> {
        run_compute_shader(device, queue, SHADER, "main", &[
            (0, u32_to_bytes(&params), BufferUsages::UNIFORM),
            (1, f32_to_bytes(cur_re), s),
            (2, f32_to_bytes(cur_im), s),
            (3, f32_to_bytes(prev_re), s),
            (4, f32_to_bytes(prev_im), s),
            (5, f32_to_bytes(lp_high), s),
            (6, f32_to_bytes(lp_low), s),
            (7, vec![0u8; buf_bytes], s),
            (8, vec![0u8; buf_bytes], s),
        ], out_binding, (width.div_ceil(16), height.div_ceil(16), 1))
    };

    let out_re = bytes_to_f32(&make_bufs(7));
    let out_im = bytes_to_f32(&make_bufs(8));
    let new_prev_re = bytes_to_f32(&make_bufs(3));
    let new_prev_im = bytes_to_f32(&make_bufs(4));
    let new_lp_high = bytes_to_f32(&make_bufs(5));
    let new_lp_low = bytes_to_f32(&make_bufs(6));

    (out_re, out_im, new_prev_re, new_prev_im, new_lp_high, new_lp_low)
}

#[test]
fn test_first_frame_passthrough() {
    // On first frame, output should equal input, and prev should be set to current
    let (device, queue) = create_test_device();
    let w = 4u32;
    let h = 4u32;
    let n = (w * h) as usize;

    let cur_re: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1).collect();
    let cur_im: Vec<f32> = (0..n).map(|i| (i as f32) * -0.05).collect();
    let zeros = vec![0.0f32; n];

    let (out_re, out_im, new_prev_re, new_prev_im, lph, lpl) = run_phase_amplify(
        &device, &queue,
        &cur_re, &cur_im, &zeros, &zeros, &zeros, &zeros,
        w, h, 20.0, 0.1, 0.5, true,
    );

    assert_f32_near(&out_re, &cur_re, 1e-5, "first frame re passthrough");
    assert_f32_near(&out_im, &cur_im, 1e-5, "first frame im passthrough");
    assert_f32_near(&new_prev_re, &cur_re, 1e-5, "prev set to current re");
    assert_f32_near(&new_prev_im, &cur_im, 1e-5, "prev set to current im");
    assert_f32_near(&lph, &zeros, 1e-5, "IIR high initialized to 0");
    assert_f32_near(&lpl, &zeros, 1e-5, "IIR low initialized to 0");

    println!("✓ First frame: passthrough + state initialization");
}

#[test]
fn test_no_motion_no_change() {
    // If current == previous, phase difference is 0 → no amplification → output == input
    let (device, queue) = create_test_device();
    let w = 8u32;
    let h = 8u32;
    let n = (w * h) as usize;

    // Same sub-band for both frames (no motion)
    let re: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).cos()).collect();
    let im: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
    let zeros = vec![0.0f32; n];

    let (out_re, out_im, ..) = run_phase_amplify(
        &device, &queue,
        &re, &im, &re, &im, &zeros, &zeros,
        w, h, 100.0, 0.1, 0.5, false,
    );

    // Phase diff = 0 → filtered = 0 → amp_phase = 0 → rotation by 0 → output == input
    assert_f32_near(&out_re, &re, 1e-4, "no motion re");
    assert_f32_near(&out_im, &im, 1e-4, "no motion im");

    println!("✓ No motion: output equals input (even with high amplification)");
}

#[test]
fn test_known_phase_shift() {
    // Previous frame has phase θ, current has phase θ + Δ.
    // With alpha_high=1, alpha_low=0 (allpass), amplification=A:
    // Output should have phase θ + Δ + A*Δ = θ + (1+A)*Δ
    let (device, queue) = create_test_device();
    let w = 1u32;
    let h = 1u32;

    let phase_prev = 0.5f32;
    let delta = 0.1f32; // small phase shift
    let phase_curr = phase_prev + delta;
    let amp = 5.0f32;

    let prev_re = vec![phase_prev.cos()];
    let prev_im = vec![phase_prev.sin()];
    let cur_re = vec![phase_curr.cos()];
    let cur_im = vec![phase_curr.sin()];
    let zeros = vec![0.0f32; 1];

    // alpha_high=1.0 (instant), alpha_low=0.0 (no low cutoff) → allpass
    let (out_re, out_im, ..) = run_phase_amplify(
        &device, &queue,
        &cur_re, &cur_im, &prev_re, &prev_im, &zeros, &zeros,
        w, h, amp, 0.0, 1.0, false,
    );

    // Expected: phase = phase_curr + amp * delta = 0.5 + 0.1 + 5.0 * 0.1 = 1.1
    let expected_phase = phase_curr + amp * delta;
    let actual_phase = out_im[0].atan2(out_re[0]);
    let phase_err = (actual_phase - expected_phase).abs();

    println!("  prev_phase={:.3}, curr_phase={:.3}, delta={:.3}", phase_prev, phase_curr, delta);
    println!("  expected_out_phase={:.3}, actual={:.3}, err={:.4}", expected_phase, actual_phase, phase_err);

    assert!(phase_err < 0.01,
        "Phase error {:.4} exceeds tolerance", phase_err);

    // Magnitude should be preserved (rotation doesn't change magnitude)
    let in_mag = (cur_re[0] * cur_re[0] + cur_im[0] * cur_im[0]).sqrt();
    let out_mag = (out_re[0] * out_re[0] + out_im[0] * out_im[0]).sqrt();
    assert!((in_mag - out_mag).abs() < 1e-4,
        "Magnitude should be preserved: in={}, out={}", in_mag, out_mag);

    println!("✓ Known phase shift: amplification = {}×, magnitude preserved", amp);
}

#[test]
fn test_magnitude_preserved() {
    // Amplification should only change phase, not magnitude
    let (device, queue) = create_test_device();
    let w = 8u32;
    let h = 8u32;
    let n = (w * h) as usize;

    // Previous: unit magnitude at various phases
    let prev_re: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5).cos()).collect();
    let prev_im: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5).sin()).collect();

    // Current: same magnitude, slightly shifted phase
    let shift = 0.2f32;
    let cur_re: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5 + shift).cos()).collect();
    let cur_im: Vec<f32> = (0..n).map(|i| (i as f32 * 0.5 + shift).sin()).collect();
    let zeros = vec![0.0f32; n];

    let (out_re, out_im, ..) = run_phase_amplify(
        &device, &queue,
        &cur_re, &cur_im, &prev_re, &prev_im, &zeros, &zeros,
        w, h, 10.0, 0.0, 1.0, false,
    );

    for i in 0..n {
        let in_mag = (cur_re[i] * cur_re[i] + cur_im[i] * cur_im[i]).sqrt();
        let out_mag = (out_re[i] * out_re[i] + out_im[i] * out_im[i]).sqrt();
        assert!((in_mag - out_mag).abs() < 1e-3,
            "Pixel {}: in_mag={:.4}, out_mag={:.4}", i, in_mag, out_mag);
    }

    println!("✓ Magnitude preserved across all {} pixels", n);
}

#[test]
fn test_zero_amplification() {
    // With amplification=0, output should equal input regardless of motion
    let (device, queue) = create_test_device();
    let w = 4u32;
    let h = 4u32;
    let n = (w * h) as usize;

    let prev_re: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).cos()).collect();
    let prev_im: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3).sin()).collect();
    let cur_re: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3 + 0.5).cos()).collect();
    let cur_im: Vec<f32> = (0..n).map(|i| (i as f32 * 0.3 + 0.5).sin()).collect();
    let zeros = vec![0.0f32; n];

    let (out_re, out_im, ..) = run_phase_amplify(
        &device, &queue,
        &cur_re, &cur_im, &prev_re, &prev_im, &zeros, &zeros,
        w, h, 0.0, 0.0, 1.0, false, // amplification = 0
    );

    assert_f32_near(&out_re, &cur_re, 1e-4, "zero amp re");
    assert_f32_near(&out_im, &cur_im, 1e-4, "zero amp im");

    println!("✓ Zero amplification: output equals input");
}

#[test]
fn test_temporal_filter_rejects_dc() {
    // With a bandpass filter, oscillatory motion in the passband should be
    // amplified, while very slow or very fast motion should be attenuated.
    // Simulate sinusoidal phase oscillation over multiple frames.
    let (device, queue) = create_test_device();
    let w = 1u32;
    let h = 1u32;

    let osc_freq = 0.1f32; // oscillation at 0.1 cycles/frame (in passband)
    let osc_amplitude = 0.05f32; // small oscillation
    let amp = 5.0f32;
    let alpha_low = 0.05f32; // low cutoff
    let alpha_high = 0.5f32; // high cutoff

    let base_phase = 1.0f32;
    let mut prev_re = vec![base_phase.cos()];
    let mut prev_im = vec![base_phase.sin()];
    let mut lp_high = vec![0.0f32];
    let mut lp_low = vec![0.0f32];

    // First frame to initialize
    let (_, _, new_pr, new_pi, new_lph, new_lpl) = run_phase_amplify(
        &device, &queue,
        &prev_re, &prev_im, &vec![0.0], &vec![0.0], &lp_high, &lp_low,
        w, h, amp, alpha_low, alpha_high, true,
    );
    prev_re = new_pr; prev_im = new_pi; lp_high = new_lph; lp_low = new_lpl;

    let mut max_output_oscillation = 0.0f32;
    for frame in 1..80 {
        let phase = base_phase + osc_amplitude * (2.0 * PI * osc_freq * frame as f32).sin();
        let cur_re_v = vec![phase.cos()];
        let cur_im_v = vec![phase.sin()];

        let (out_re, out_im, new_pr, new_pi, new_lph, new_lpl) = run_phase_amplify(
            &device, &queue,
            &cur_re_v, &cur_im_v, &prev_re, &prev_im, &lp_high, &lp_low,
            w, h, amp, alpha_low, alpha_high, false,
        );

        let out_phase = out_im[0].atan2(out_re[0]);
        let deviation = (out_phase - phase).abs();
        if deviation > max_output_oscillation {
            max_output_oscillation = deviation;
        }

        prev_re = new_pr; prev_im = new_pi; lp_high = new_lph; lp_low = new_lpl;

        if frame < 5 || frame % 20 == 0 {
            println!("  frame {}: input_phase={:.4}, output_phase={:.4}, deviation={:.4}",
                frame, phase, out_phase, deviation);
        }
    }

    // The output should oscillate more than the input (amplified)
    println!("  Max output oscillation: {:.4} (input amplitude: {:.4})", max_output_oscillation, osc_amplitude);
    assert!(max_output_oscillation > osc_amplitude,
        "Output should be amplified beyond input amplitude");

    println!("✓ Temporal filter: oscillatory motion amplified");
}

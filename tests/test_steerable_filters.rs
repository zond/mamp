//! Tests for steerable_filters.wgsl — steerable pyramid filter bank
//!
//! Run: cargo test --features native-test --test test_steerable_filters -- --nocapture

#[path = "gpu_test_harness.rs"]
mod gpu_test_harness;
use gpu_test_harness::*;

use std::f32::consts::PI;
use wgpu::BufferUsages;

const SHADER: &str = include_str!("../src/shaders/steerable_filters.wgsl");

/// Apply a steerable filter to a spectrum and return the filtered spectrum.
fn apply_filter(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input_re: &[f32],
    input_im: &[f32],
    width: u32,
    height: u32,
    num_orientations: u32,
    orientation_idx: u32,
    scale: u32,
    num_scales: u32,
    filter_type: u32,
) -> (Vec<f32>, Vec<f32>) {
    let params: [u32; 8] = [
        width, height, num_orientations, orientation_idx,
        scale, num_scales, filter_type, 0,
    ];
    let total = (width * height) as usize;
    let buf_bytes = total * 4;

    let re = run_compute_shader(
        device, queue, SHADER, "main",
        &[
            (0, u32_to_bytes(&params), BufferUsages::UNIFORM),
            (1, f32_to_bytes(input_re), BufferUsages::STORAGE),
            (2, f32_to_bytes(input_im), BufferUsages::STORAGE),
            (3, vec![0u8; buf_bytes], BufferUsages::STORAGE),
            (4, vec![0u8; buf_bytes], BufferUsages::STORAGE),
        ],
        3,
        (width.div_ceil(16), height.div_ceil(16), 1),
    );
    let im = run_compute_shader(
        device, queue, SHADER, "main",
        &[
            (0, u32_to_bytes(&params), BufferUsages::UNIFORM),
            (1, f32_to_bytes(input_re), BufferUsages::STORAGE),
            (2, f32_to_bytes(input_im), BufferUsages::STORAGE),
            (3, vec![0u8; buf_bytes], BufferUsages::STORAGE),
            (4, vec![0u8; buf_bytes], BufferUsages::STORAGE),
        ],
        4,
        (width.div_ceil(16), height.div_ceil(16), 1),
    );

    (bytes_to_f32(&re), bytes_to_f32(&im))
}

/// Compute the filter magnitude (not the filtered output — apply to unit spectrum)
fn get_filter_magnitude(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    num_orientations: u32,
    orientation_idx: u32,
    scale: u32,
    num_scales: u32,
    filter_type: u32,
) -> Vec<f32> {
    let total = (width * height) as usize;
    // Input: unit spectrum (all ones)
    let ones = vec![1.0f32; total];
    let zeros = vec![0.0f32; total];

    let (re, im) = apply_filter(
        device, queue, &ones, &zeros,
        width, height, num_orientations, orientation_idx,
        scale, num_scales, filter_type,
    );

    // Magnitude = sqrt(re^2 + im^2). Since input was (1, 0), output = (filter, 0),
    // so magnitude = |re| (im should be 0)
    re.iter().map(|r| r.abs()).collect()
}

#[test]
fn test_filters_are_nonzero() {
    // Each filter type should produce nonzero output somewhere
    let (device, queue) = create_test_device();
    let w = 32u32;
    let h = 32u32;
    let n_orient = 4u32;
    let n_scales = 3u32;

    // Highpass residual
    let hi = get_filter_magnitude(&device, &queue, w, h, n_orient, 0, 0, n_scales, 1);
    assert!(hi.iter().any(|&v| v > 0.1), "Highpass should be nonzero somewhere");

    // Lowpass residual
    let lo = get_filter_magnitude(&device, &queue, w, h, n_orient, 0, 0, n_scales, 2);
    assert!(lo.iter().any(|&v| v > 0.1), "Lowpass should be nonzero somewhere");

    // Bandpass sub-bands
    for s in 0..n_scales {
        for o in 0..n_orient {
            let bp = get_filter_magnitude(&device, &queue, w, h, n_orient, o, s, n_scales, 0);
            assert!(bp.iter().any(|&v| v > 0.01),
                "Bandpass scale={} orient={} should be nonzero", s, o);
        }
    }

    println!("✓ All filters produce nonzero output");
}

#[test]
fn test_filter_dc_blocked() {
    // DC (0,0) should only pass through the lowpass residual
    let (device, queue) = create_test_device();
    let w = 32u32;
    let h = 32u32;

    // Highpass at DC
    let hi = get_filter_magnitude(&device, &queue, w, h, 4, 0, 0, 3, 1);
    assert!(hi[0] < 0.01, "Highpass should block DC, got {}", hi[0]);

    // Bandpass at DC
    let bp = get_filter_magnitude(&device, &queue, w, h, 4, 0, 0, 3, 0);
    assert!(bp[0] < 0.01, "Bandpass should block DC, got {}", bp[0]);

    // Lowpass at DC
    let lo = get_filter_magnitude(&device, &queue, w, h, 4, 0, 0, 3, 2);
    assert!(lo[0] > 0.5, "Lowpass should pass DC, got {}", lo[0]);

    println!("✓ DC correctly routed to lowpass only");
}

#[test]
fn test_filter_orientation_selectivity() {
    // A horizontal grating (fy=0, fx≠0) should be strongest in the orientation
    // closest to horizontal (angle=0)
    let (device, queue) = create_test_device();
    let w = 64u32;
    let h = 64u32;
    let n_orient = 4u32;
    let n_scales = 3u32;

    // Put energy at (fx=w/4, fy=0) — horizontal, at scale 0 (finest).
    // Scale 0 center_log_rad = -1, i.e. radius = 0.5. fx = w/4 → radius = (w/4)/w = 0.25.
    // log2(0.25) = -2, which is scale 1. Use fx = w/2 * 0.7 ≈ radius 0.35 (between -1 and -2).
    let total = (w * h) as usize;
    let mut spec_re = vec![0.0f32; total];
    let mut spec_im = vec![0.0f32; total];
    // fx near Nyquist/2 for finest scale
    let fx_idx = 28usize; // radius = 28/64 = 0.4375, log2 ≈ -1.19 → well within scale 0
    spec_re[fx_idx] = 1.0;

    // Debug: check what the filter looks like at various frequencies along the x-axis (fy=0)
    let filt_mag = get_filter_magnitude(&device, &queue, w, h, n_orient, 0, 0, n_scales, 0);
    println!("  Scale 0, Orient 0 filter along x-axis:");
    for fx in 0..w as usize {
        if filt_mag[fx] > 0.001 {
            let radius = fx as f32 / w as f32;
            println!("    fx={}: mag={:.4}, radius={:.3}, log2r={:.2}", fx, filt_mag[fx], radius, radius.log2());
        }
    }

    let mut energies = vec![0.0f32; n_orient as usize];
    for o in 0..n_orient {
        let (filt_re, filt_im) = apply_filter(
            &device, &queue, &spec_re, &spec_im,
            w, h, n_orient, o, 0, n_scales, 0,
        );
        let energy: f32 = filt_re.iter().zip(filt_im.iter())
            .map(|(r, i)| r * r + i * i).sum();
        energies[o as usize] = energy;
    }

    // Orientation 0 (0°, horizontal) should have highest energy
    let max_orient = energies.iter().enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap().0;

    println!("  Orientation energies for horizontal grating: {:?}", energies);
    assert_eq!(max_orient, 0,
        "Horizontal grating should be strongest at orientation 0, got {}", max_orient);
    println!("✓ Orientation selectivity: horizontal → orientation 0");
}

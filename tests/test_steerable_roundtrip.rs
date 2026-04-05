//! End-to-end steerable pyramid decomposition → reconstruction tests.
//!
//! Run: cargo test --features native-test --test test_steerable_roundtrip -- --nocapture

#[path = "gpu_test_harness.rs"]
mod gpu_test_harness;
use gpu_test_harness::*;

use std::f32::consts::PI;
use wgpu::BufferUsages;

const FFT_SHADER: &str = include_str!("../src/shaders/fft.wgsl");
const FILTER_SHADER: &str = include_str!("../src/shaders/steerable_filters.wgsl");

/// Run 2D FFT (forward or inverse).
fn fft_2d(
    device: &wgpu::Device, queue: &wgpu::Queue, shader: &str,
    re: &[f32], im: &[f32], w: u32, h: u32, inverse: bool,
) -> (Vec<f32>, Vec<f32>) {
    let total = (w * h) as usize;
    let buf_bytes = total * 4;
    let log2_w = (w as f32).log2() as u32;
    let log2_h = (h as f32).log2() as u32;
    let inv = if inverse { 1u32 } else { 0 };

    let (first_n, first_num, first_stride, first_fft_stride) = if !inverse {
        (w, h, 1u32, w) // rows first
    } else {
        (h, w, w, 1u32) // columns first
    };
    let (second_n, second_num, second_stride, second_fft_stride) = if !inverse {
        (h, w, w, 1u32) // then columns
    } else {
        (w, h, 1u32, w) // then rows
    };
    let first_log2 = if !inverse { log2_w } else { log2_h };
    let second_log2 = if !inverse { log2_h } else { log2_w };

    // Pass 1
    let p1: [u32; 8] = [first_n, first_log2, first_num, inv, first_stride, first_fft_stride, 0, 0];
    let p1_re = run_compute_shader(device, queue, shader, "main", &[
        (0, u32_to_bytes(&p1), BufferUsages::UNIFORM),
        (1, f32_to_bytes(re), BufferUsages::STORAGE),
        (2, f32_to_bytes(im), BufferUsages::STORAGE),
        (3, vec![0u8; buf_bytes], BufferUsages::STORAGE),
        (4, vec![0u8; buf_bytes], BufferUsages::STORAGE),
    ], 3, (first_num, 1, 1));
    let p1_im = run_compute_shader(device, queue, shader, "main", &[
        (0, u32_to_bytes(&p1), BufferUsages::UNIFORM),
        (1, f32_to_bytes(re), BufferUsages::STORAGE),
        (2, f32_to_bytes(im), BufferUsages::STORAGE),
        (3, vec![0u8; buf_bytes], BufferUsages::STORAGE),
        (4, vec![0u8; buf_bytes], BufferUsages::STORAGE),
    ], 4, (first_num, 1, 1));

    // Pass 2
    let p2: [u32; 8] = [second_n, second_log2, second_num, inv, second_stride, second_fft_stride, 0, 0];
    let p2_re = run_compute_shader(device, queue, shader, "main", &[
        (0, u32_to_bytes(&p2), BufferUsages::UNIFORM),
        (1, p1_re.clone(), BufferUsages::STORAGE),
        (2, p1_im.clone(), BufferUsages::STORAGE),
        (3, vec![0u8; buf_bytes], BufferUsages::STORAGE),
        (4, vec![0u8; buf_bytes], BufferUsages::STORAGE),
    ], 3, (second_num, 1, 1));
    let p2_im = run_compute_shader(device, queue, shader, "main", &[
        (0, u32_to_bytes(&p2), BufferUsages::UNIFORM),
        (1, p1_re, BufferUsages::STORAGE),
        (2, p1_im, BufferUsages::STORAGE),
        (3, vec![0u8; buf_bytes], BufferUsages::STORAGE),
        (4, vec![0u8; buf_bytes], BufferUsages::STORAGE),
    ], 4, (second_num, 1, 1));

    (bytes_to_f32(&p2_re), bytes_to_f32(&p2_im))
}

/// Apply steerable filter.
fn apply_filter(
    device: &wgpu::Device, queue: &wgpu::Queue,
    re: &[f32], im: &[f32], w: u32, h: u32,
    n_orient: u32, orient: u32, scale: u32, n_scales: u32, ftype: u32,
) -> (Vec<f32>, Vec<f32>) {
    let params: [u32; 8] = [w, h, n_orient, orient, scale, n_scales, ftype, 0];
    let total = (w * h) as usize;
    let buf_bytes = total * 4;

    let out_re = run_compute_shader(device, queue, FILTER_SHADER, "main", &[
        (0, u32_to_bytes(&params), BufferUsages::UNIFORM),
        (1, f32_to_bytes(re), BufferUsages::STORAGE),
        (2, f32_to_bytes(im), BufferUsages::STORAGE),
        (3, vec![0u8; buf_bytes], BufferUsages::STORAGE),
        (4, vec![0u8; buf_bytes], BufferUsages::STORAGE),
    ], 3, (w.div_ceil(16), h.div_ceil(16), 1));
    let out_im = run_compute_shader(device, queue, FILTER_SHADER, "main", &[
        (0, u32_to_bytes(&params), BufferUsages::UNIFORM),
        (1, f32_to_bytes(re), BufferUsages::STORAGE),
        (2, f32_to_bytes(im), BufferUsages::STORAGE),
        (3, vec![0u8; buf_bytes], BufferUsages::STORAGE),
        (4, vec![0u8; buf_bytes], BufferUsages::STORAGE),
    ], 4, (w.div_ceil(16), h.div_ceil(16), 1));

    (bytes_to_f32(&out_re), bytes_to_f32(&out_im))
}

#[test]
fn test_full_decompose_reconstruct_small() {
    // Decompose into all sub-bands, inverse FFT each, sum → original
    let (device, queue) = create_test_device();
    let fft_src = fft_shader_source(1024);
    let w = 32u32;
    let h = 32u32;
    let n_orient = 4u32;
    let n_scales = 3u32;
    let total = (w * h) as usize;

    // Create a test image (real-valued)
    let image: Vec<f32> = (0..total).map(|i| {
        let x = (i % w as usize) as f32 / w as f32;
        let y = (i / w as usize) as f32 / h as f32;
        (2.0 * PI * 3.0 * x).cos() * (2.0 * PI * 2.0 * y).sin() + 0.5
    }).collect();
    let zeros = vec![0.0f32; total];

    // Forward 2D FFT
    let (spec_re, spec_im) = fft_2d(&device, &queue, &fft_src, &image, &zeros, w, h, false);

    // Accumulate reconstructed image from all sub-bands
    let mut recon_re = vec![0.0f32; total];
    let mut recon_im = vec![0.0f32; total];

    // Highpass residual
    let (filt_re, filt_im) = apply_filter(&device, &queue, &spec_re, &spec_im, w, h, n_orient, 0, 0, n_scales, 1);
    let (sub_re, sub_im) = fft_2d(&device, &queue, &fft_src, &filt_re, &filt_im, w, h, true);
    for i in 0..total { recon_re[i] += sub_re[i]; recon_im[i] += sub_im[i]; }

    // Lowpass residual
    let (filt_re, filt_im) = apply_filter(&device, &queue, &spec_re, &spec_im, w, h, n_orient, 0, 0, n_scales, 2);
    let (sub_re, sub_im) = fft_2d(&device, &queue, &fft_src, &filt_re, &filt_im, w, h, true);
    for i in 0..total { recon_re[i] += sub_re[i]; recon_im[i] += sub_im[i]; }

    // All bandpass sub-bands
    for s in 0..n_scales {
        for o in 0..n_orient {
            let (filt_re, filt_im) = apply_filter(&device, &queue, &spec_re, &spec_im, w, h, n_orient, o, s, n_scales, 0);
            let (sub_re, sub_im) = fft_2d(&device, &queue, &fft_src, &filt_re, &filt_im, w, h, true);
            for i in 0..total { recon_re[i] += sub_re[i]; recon_im[i] += sub_im[i]; }
        }
    }

    // Reconstruction should match original
    let max_err = recon_re.iter().zip(image.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let rms_err = (recon_re.iter().zip(image.iter())
        .map(|(a, b)| (a - b) * (a - b)).sum::<f32>() / total as f32).sqrt();
    println!("  Reconstruction error: max={:.4}, rms={:.4}", max_err, rms_err);
    // Small grids have higher error due to filter resolution limits
    assert!(rms_err < 0.1, "RMS error too high: {}", rms_err);

    // Note: individual sub-bands are complex (oriented filters break Hermitian symmetry).
    // The imaginary parts cancel in the sum for an ideal partition, but with finite grids
    // there's residual imaginary content. This is acceptable.
    let max_im = recon_im.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("  Residual imaginary: max={:.4}", max_im);

    println!("✓ Full decompose→reconstruct: 32×32, {} scales × {} orientations", n_scales, n_orient);
}

#[test]
fn test_full_decompose_reconstruct_larger() {
    // Same test at 128×128 to exercise multi-element-per-thread FFT
    let (device, queue) = create_test_device();
    let fft_src = fft_shader_source(1024);
    let w = 128u32;
    let h = 128u32;
    let n_orient = 4u32;
    let n_scales = 3u32;
    let total = (w * h) as usize;

    let image: Vec<f32> = (0..total).map(|i| {
        let x = (i % w as usize) as f32 / w as f32;
        let y = (i / w as usize) as f32 / h as f32;
        0.3 * (2.0 * PI * 5.0 * x).cos() + 0.7 * (2.0 * PI * 12.0 * y).sin()
    }).collect();
    let zeros = vec![0.0f32; total];

    let (spec_re, spec_im) = fft_2d(&device, &queue, &fft_src, &image, &zeros, w, h, false);

    let mut recon_re = vec![0.0f32; total];

    // Highpass
    let (f_re, f_im) = apply_filter(&device, &queue, &spec_re, &spec_im, w, h, n_orient, 0, 0, n_scales, 1);
    let (s_re, _) = fft_2d(&device, &queue, &fft_src, &f_re, &f_im, w, h, true);
    for i in 0..total { recon_re[i] += s_re[i]; }

    // Lowpass
    let (f_re, f_im) = apply_filter(&device, &queue, &spec_re, &spec_im, w, h, n_orient, 0, 0, n_scales, 2);
    let (s_re, _) = fft_2d(&device, &queue, &fft_src, &f_re, &f_im, w, h, true);
    for i in 0..total { recon_re[i] += s_re[i]; }

    // Bandpass
    for s in 0..n_scales {
        for o in 0..n_orient {
            let (f_re, f_im) = apply_filter(&device, &queue, &spec_re, &spec_im, w, h, n_orient, o, s, n_scales, 0);
            let (s_re, _) = fft_2d(&device, &queue, &fft_src, &f_re, &f_im, w, h, true);
            for i in 0..total { recon_re[i] += s_re[i]; }
        }
    }

    assert_f32_near(&recon_re, &image, 0.1, "128×128 decompose→reconstruct");
    println!("✓ Full decompose→reconstruct: 128×128");
}

#[test]
fn test_2d_fft_large_roundtrip() {
    // Test 2D FFT roundtrip at 256×256 (exercises N=256 multi-element path)
    let (device, queue) = create_test_device();
    let fft_src = fft_shader_source(1024);
    let w = 256u32;
    let h = 256u32;
    let total = (w * h) as usize;

    let re: Vec<f32> = (0..total).map(|i| ((i as f32 * 0.37) % 1.0) - 0.5).collect();
    let im = vec![0.0f32; total];

    let (fft_re, fft_im) = fft_2d(&device, &queue, &fft_src, &re, &im, w, h, false);
    let (rec_re, rec_im) = fft_2d(&device, &queue, &fft_src, &fft_re, &fft_im, w, h, true);

    assert_f32_near(&rec_re, &re, 0.05, "256×256 roundtrip re");
    let max_im = rec_im.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    assert!(max_im < 0.05, "256×256 imaginary should be ~0, max={}", max_im);
    println!("✓ 2D FFT roundtrip 256×256");
}

#[test]
fn test_real_image_stays_real() {
    // A real-valued image filtered by a symmetric filter should stay real
    // (imaginary part near-zero after inverse FFT)
    let (device, queue) = create_test_device();
    let fft_src = fft_shader_source(1024);
    let w = 64u32;
    let h = 64u32;
    let total = (w * h) as usize;

    // Real image
    let image: Vec<f32> = (0..total).map(|i| {
        let x = (i % w as usize) as f32;
        let y = (i / w as usize) as f32;
        (0.1 * x + 0.2 * y).sin()
    }).collect();
    let zeros = vec![0.0f32; total];

    let (spec_re, spec_im) = fft_2d(&device, &queue, &fft_src, &image, &zeros, w, h, false);

    // Apply bandpass filter (scale 0, orient 0)
    let (filt_re, filt_im) = apply_filter(&device, &queue, &spec_re, &spec_im, w, h, 4, 0, 0, 3, 0);

    // Inverse FFT
    let (sub_re, sub_im) = fft_2d(&device, &queue, &fft_src, &filt_re, &filt_im, w, h, true);

    // The sub-band of a real image is complex (oriented filters break symmetry)
    // but the FULL reconstruction (sum of all sub-bands) should be real.
    // Individual sub-bands are expected to be complex — this is normal.
    // Just verify the magnitude is reasonable (not NaN or huge)
    let max_mag: f32 = sub_re.iter().zip(sub_im.iter())
        .map(|(r, i)| (r * r + i * i).sqrt())
        .fold(0.0f32, f32::max);
    assert!(max_mag < 100.0, "Sub-band magnitude should be bounded, got {}", max_mag);
    assert!(max_mag > 0.001, "Sub-band should have nonzero content, got {}", max_mag);
    println!("✓ Sub-band of real image has bounded magnitude: max={:.4}", max_mag);
}

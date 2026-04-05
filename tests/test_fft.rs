//! Tests for fft.wgsl — 1D FFT compute shader
//!
//! Tests both the 2048-point (16KB shared) and 1024-point (8KB fallback) variants.
//! Run: cargo test --features native-test test_fft -- --nocapture

#[path = "gpu_test_harness.rs"]
mod gpu_test_harness;
use gpu_test_harness::*;

use std::f32::consts::PI;
use wgpu::BufferUsages;

/// Run a 1D FFT on the GPU. Uses stride=1, fft_stride=n (contiguous rows).
fn run_fft(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shader_src: &str,
    input_re: &[f32],
    input_im: &[f32],
    n: u32,
    num_ffts: u32,
    inverse: bool,
) -> (Vec<f32>, Vec<f32>) {
    let log2_n = (n as f32).log2() as u32;
    assert_eq!(1u32 << log2_n, n, "n must be power of 2");

    // Params: n, log2_n, num_ffts, inverse, stride, fft_stride, pad, pad
    let params: [u32; 8] = [n, log2_n, num_ffts, if inverse { 1 } else { 0 }, 1, n, 0, 0];
    let total = (n * num_ffts) as usize;
    let buf_bytes = total * 4;

    // Run for output_re (binding 3)
    let result_re = run_compute_shader(
        device, queue, shader_src, "main",
        &[
            (0, u32_to_bytes(&params), BufferUsages::UNIFORM),
            (1, f32_to_bytes(input_re), BufferUsages::STORAGE),
            (2, f32_to_bytes(input_im), BufferUsages::STORAGE),
            (3, vec![0u8; buf_bytes], BufferUsages::STORAGE),
            (4, vec![0u8; buf_bytes], BufferUsages::STORAGE),
        ],
        3,
        (num_ffts, 1, 1),
    );

    // Run for output_im (binding 4)
    let result_im = run_compute_shader(
        device, queue, shader_src, "main",
        &[
            (0, u32_to_bytes(&params), BufferUsages::UNIFORM),
            (1, f32_to_bytes(input_re), BufferUsages::STORAGE),
            (2, f32_to_bytes(input_im), BufferUsages::STORAGE),
            (3, vec![0u8; buf_bytes], BufferUsages::STORAGE),
            (4, vec![0u8; buf_bytes], BufferUsages::STORAGE),
        ],
        4,
        (num_ffts, 1, 1),
    );

    (bytes_to_f32(&result_re), bytes_to_f32(&result_im))
}

/// CPU reference DFT.
fn cpu_dft(re: &[f32], im: &[f32], inverse: bool) -> (Vec<f32>, Vec<f32>) {
    let n = re.len();
    let sign = if inverse { 1.0 } else { -1.0 };
    let scale = if inverse { 1.0 / n as f32 } else { 1.0 };
    let mut out_re = vec![0.0f32; n];
    let mut out_im = vec![0.0f32; n];
    for k in 0..n {
        let (mut sr, mut si) = (0.0f32, 0.0f32);
        for j in 0..n {
            let angle = sign * 2.0 * PI * (k as f32) * (j as f32) / (n as f32);
            let (c, s) = (angle.cos(), angle.sin());
            sr += re[j] * c - im[j] * s;
            si += re[j] * s + im[j] * c;
        }
        out_re[k] = sr * scale;
        out_im[k] = si * scale;
    }
    (out_re, out_im)
}

// ── Tests using 1024-point shader (guaranteed to work everywhere) ──

#[test]
fn test_fft_1024_delta() {
    let (device, queue) = create_test_device();
    let src = fft_shader_source(1024);
    let n = 16u32;
    let mut re = vec![0.0f32; n as usize];
    re[0] = 1.0;
    let im = vec![0.0f32; n as usize];

    let (out_re, out_im) = run_fft(&device, &queue, &src, &re, &im, n, 1, false);
    assert_f32_near(&out_re, &vec![1.0; n as usize], 1e-4, "delta re");
    assert_f32_near(&out_im, &vec![0.0; n as usize], 1e-4, "delta im");
    println!("✓ FFT-1024 delta N={}", n);
}

#[test]
fn test_fft_1024_roundtrip_256() {
    let (device, queue) = create_test_device();
    let src = fft_shader_source(1024);
    let n = 256u32;
    let re: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin()).collect();
    let im: Vec<f32> = (0..n).map(|i| (i as f32 * 0.07).cos()).collect();

    let (fft_re, fft_im) = run_fft(&device, &queue, &src, &re, &im, n, 1, false);
    let (rec_re, rec_im) = run_fft(&device, &queue, &src, &fft_re, &fft_im, n, 1, true);

    assert_f32_near(&rec_re, &re, 1e-2, "roundtrip re");
    assert_f32_near(&rec_im, &im, 1e-2, "roundtrip im");
    println!("✓ FFT-1024 roundtrip N={}", n);
}

#[test]
fn test_fft_1024_vs_cpu_512() {
    let (device, queue) = create_test_device();
    let src = fft_shader_source(1024);
    let n = 512u32;
    let re: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.618) % 1.0) - 0.5).collect();
    let im: Vec<f32> = (0..n).map(|i| ((i as f32 * 0.414) % 1.0) - 0.5).collect();

    let (gpu_re, gpu_im) = run_fft(&device, &queue, &src, &re, &im, n, 1, false);
    let (cpu_re, cpu_im) = cpu_dft(&re, &im, false);

    assert_f32_near(&gpu_re, &cpu_re, 0.1, "vs cpu re");
    assert_f32_near(&gpu_im, &cpu_im, 0.1, "vs cpu im");
    println!("✓ FFT-1024 matches CPU DFT N={}", n);
}

#[test]
fn test_fft_1024_at_max() {
    let (device, queue) = create_test_device();
    let src = fft_shader_source(1024);
    let n = 1024u32;

    // Cosine at known frequency
    let freq = 42usize;
    let re: Vec<f32> = (0..n).map(|i| (2.0 * PI * freq as f32 * i as f32 / n as f32).cos()).collect();
    let im = vec![0.0f32; n as usize];

    let (fft_re, fft_im) = run_fft(&device, &queue, &src, &re, &im, n, 1, false);
    let mag: Vec<f32> = fft_re.iter().zip(fft_im.iter()).map(|(r, i)| (r*r + i*i).sqrt()).collect();

    let expected_peak = n as f32 / 2.0;
    assert!((mag[freq] - expected_peak).abs() < 1.0,
        "peak at {}: expected {}, got {}", freq, expected_peak, mag[freq]);
    assert!((mag[n as usize - freq] - expected_peak).abs() < 1.0,
        "mirror peak: expected {}, got {}", expected_peak, mag[n as usize - freq]);

    // Roundtrip
    let (rec_re, _) = run_fft(&device, &queue, &src, &fft_re, &fft_im, n, 1, true);
    assert_f32_near(&rec_re, &re, 0.05, "N=1024 roundtrip");
    println!("✓ FFT-1024 cosine + roundtrip at N=1024");
}

// ── Tests using 2048-point shader (needs 16KB shared memory) ──

#[test]
fn test_fft_2048_if_supported() {
    let (device, queue) = create_test_device();
    let max_n = max_fft_length(&device);
    println!("Device max FFT length: {} (shared memory: {} bytes)", max_n, max_shared_memory(&device));

    if max_n < 2048 {
        println!("⚠ Skipping N=2048 test — device only supports N={}", max_n);
        return;
    }

    let src = fft_shader_source(2048);
    let n = 2048u32;

    // Cosine roundtrip
    let freq = 100usize;
    let re: Vec<f32> = (0..n).map(|i| (2.0 * PI * freq as f32 * i as f32 / n as f32).cos()).collect();
    let im = vec![0.0f32; n as usize];

    let (fft_re, fft_im) = run_fft(&device, &queue, &src, &re, &im, n, 1, false);

    let mag: Vec<f32> = fft_re.iter().zip(fft_im.iter()).map(|(r, i)| (r*r + i*i).sqrt()).collect();
    let expected = n as f32 / 2.0;
    assert!((mag[freq] - expected).abs() < 2.0,
        "N=2048 peak at {}: expected {}, got {}", freq, expected, mag[freq]);

    let (rec_re, _) = run_fft(&device, &queue, &src, &fft_re, &fft_im, n, 1, true);
    assert_f32_near(&rec_re, &re, 0.1, "N=2048 roundtrip");
    println!("✓ FFT-2048 cosine + roundtrip at N=2048");
}

#[test]
fn test_fft_2048_fallback_to_1024() {
    // Even if device supports 2048, verify 1024 variant still works
    // at sizes that would need 2048 (by processing at 1024 and checking it's valid)
    let (device, queue) = create_test_device();
    let src_1024 = fft_shader_source(1024);
    let n = 1024u32;

    // Signal that needs full 1024 bandwidth
    let re: Vec<f32> = (0..n).map(|i| {
        (2.0 * PI * 3.0 * i as f32 / n as f32).cos()
        + (2.0 * PI * 100.0 * i as f32 / n as f32).sin()
        + (2.0 * PI * 500.0 * i as f32 / n as f32).cos()
    }).collect();
    let im = vec![0.0f32; n as usize];

    let (fft_re, fft_im) = run_fft(&device, &queue, &src_1024, &re, &im, n, 1, false);
    let (rec_re, _) = run_fft(&device, &queue, &src_1024, &fft_re, &fft_im, n, 1, true);

    assert_f32_near(&rec_re, &re, 0.05, "fallback roundtrip");
    println!("✓ FFT-1024 fallback: complex signal roundtrip at N=1024");
}

#[test]
fn test_fft_1024_cosine_64() {
    // Smaller test to verify basic correctness at N=64
    let (device, queue) = create_test_device();
    let src = fft_shader_source(1024);
    let n = 64u32;
    let freq = 5usize;
    let re: Vec<f32> = (0..n).map(|i| (2.0 * PI * freq as f32 * i as f32 / n as f32).cos()).collect();
    let im = vec![0.0f32; n as usize];

    let (out_re, out_im) = run_fft(&device, &queue, &src, &re, &im, n, 1, false);
    let mag: Vec<f32> = out_re.iter().zip(out_im.iter()).map(|(r, i)| (r*r + i*i).sqrt()).collect();

    let expected = n as f32 / 2.0;
    assert!((mag[freq] - expected).abs() < 0.5, "peak at {}", freq);

    for i in 0..n as usize {
        if i != freq && i != n as usize - freq {
            assert!(mag[i] < 0.5, "bin {} should be ~0, got {}", i, mag[i]);
        }
    }
    println!("✓ FFT-1024 cosine N=64, freq={}", freq);
}

#[test]
fn test_fft_batch_1024() {
    let (device, queue) = create_test_device();
    let src = fft_shader_source(1024);
    let n = 128u32;
    let num_ffts = 3u32;

    let mut re = vec![0.0f32; (n * num_ffts) as usize];
    let im = vec![0.0f32; (n * num_ffts) as usize];

    // Row 0: delta
    re[0] = 1.0;
    // Row 1: constant 2.0
    for i in 0..n as usize { re[n as usize + i] = 2.0; }
    // Row 2: single cosine
    for i in 0..n as usize {
        re[2 * n as usize + i] = (2.0 * PI * 10.0 * i as f32 / n as f32).cos();
    }

    let (out_re, _) = run_fft(&device, &queue, &src, &re, &im, n, num_ffts, false);

    // Row 0: all ones
    assert_f32_near(&out_re[0..n as usize], &vec![1.0; n as usize], 1e-3, "batch row 0");
    // Row 1: DC = 2*N
    assert!((out_re[n as usize] - 2.0 * n as f32).abs() < 0.1, "batch row 1 DC");
    println!("✓ FFT-1024 batch: {} rows of N={}", num_ffts, n);
}

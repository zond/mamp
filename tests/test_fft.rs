//! Tests for fft.wgsl — 1D FFT compute shader
//!
//! Run: cargo test --features native-test test_fft -- --nocapture

#[path = "gpu_test_harness.rs"]
mod gpu_test_harness;
use gpu_test_harness::*;

use std::f32::consts::PI;
use wgpu::BufferUsages;

const FFT_SHADER: &str = include_str!("../src/shaders/fft.wgsl");

/// Run a 1D FFT on the GPU and return (output_re, output_im).
fn run_fft(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input_re: &[f32],
    input_im: &[f32],
    inverse: bool,
) -> (Vec<f32>, Vec<f32>) {
    let n = input_re.len() as u32;
    let log2_n = (n as f32).log2() as u32;
    assert_eq!(1 << log2_n, n, "FFT length must be power of 2");

    let params: [u32; 4] = [n, log2_n, 1, if inverse { 1 } else { 0 }];
    let buf_size = (n as usize) * 4; // f32 bytes

    let result = run_compute_shader(
        device,
        queue,
        FFT_SHADER,
        "main",
        &[
            (0, u32_to_bytes(&params), BufferUsages::UNIFORM),
            (1, f32_to_bytes(input_re), BufferUsages::STORAGE),
            (2, f32_to_bytes(input_im), BufferUsages::STORAGE),
            (3, vec![0u8; buf_size], BufferUsages::STORAGE),
            (4, vec![0u8; buf_size], BufferUsages::STORAGE),
        ],
        3, // output_re binding
        (1, 1, 1), // 1 row
    );
    let out_re = bytes_to_f32(&result);

    // Need to read output_im too — run again reading binding 4
    let result_im = run_compute_shader(
        device,
        queue,
        FFT_SHADER,
        "main",
        &[
            (0, u32_to_bytes(&params), BufferUsages::UNIFORM),
            (1, f32_to_bytes(input_re), BufferUsages::STORAGE),
            (2, f32_to_bytes(input_im), BufferUsages::STORAGE),
            (3, vec![0u8; buf_size], BufferUsages::STORAGE),
            (4, vec![0u8; buf_size], BufferUsages::STORAGE),
        ],
        4, // output_im binding
        (1, 1, 1),
    );
    let out_im = bytes_to_f32(&result_im);

    (out_re, out_im)
}

/// CPU reference FFT (DFT) for comparison.
fn cpu_dft(re: &[f32], im: &[f32], inverse: bool) -> (Vec<f32>, Vec<f32>) {
    let n = re.len();
    let sign = if inverse { 1.0 } else { -1.0 };
    let scale = if inverse { 1.0 / n as f32 } else { 1.0 };
    let mut out_re = vec![0.0f32; n];
    let mut out_im = vec![0.0f32; n];

    for k in 0..n {
        let mut sum_re = 0.0f32;
        let mut sum_im = 0.0f32;
        for j in 0..n {
            let angle = sign * 2.0 * PI * (k as f32) * (j as f32) / (n as f32);
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            sum_re += re[j] * cos_a - im[j] * sin_a;
            sum_im += re[j] * sin_a + im[j] * cos_a;
        }
        out_re[k] = sum_re * scale;
        out_im[k] = sum_im * scale;
    }
    (out_re, out_im)
}

#[test]
fn test_fft_delta() {
    // FFT of delta function [1, 0, 0, ..., 0] should be all ones
    let (device, queue) = create_test_device();
    let n = 16;
    let mut input_re = vec![0.0f32; n];
    input_re[0] = 1.0;
    let input_im = vec![0.0f32; n];

    let (out_re, out_im) = run_fft(&device, &queue, &input_re, &input_im, false);

    assert_f32_near(&out_re, &vec![1.0; n], 1e-4, "FFT(delta) real");
    assert_f32_near(&out_im, &vec![0.0; n], 1e-4, "FFT(delta) imag");
    println!("✓ FFT delta: all ones in spectrum");
}

#[test]
fn test_fft_constant() {
    // FFT of constant [1, 1, ..., 1] should be [N, 0, 0, ..., 0]
    let (device, queue) = create_test_device();
    let n = 16;
    let input_re = vec![1.0f32; n];
    let input_im = vec![0.0f32; n];

    let (out_re, out_im) = run_fft(&device, &queue, &input_re, &input_im, false);

    let mut expected_re = vec![0.0f32; n];
    expected_re[0] = n as f32;
    assert_f32_near(&out_re, &expected_re, 1e-3, "FFT(constant) real");
    assert_f32_near(&out_im, &vec![0.0; n], 1e-3, "FFT(constant) imag");
    println!("✓ FFT constant: N at DC, zeros elsewhere");
}

#[test]
fn test_fft_cosine() {
    // FFT of cos(2*pi*k*n/N) should have peaks at bins k and N-k
    let (device, queue) = create_test_device();
    let n = 64;
    let freq = 5; // 5 cycles
    let input_re: Vec<f32> = (0..n)
        .map(|i| (2.0 * PI * freq as f32 * i as f32 / n as f32).cos())
        .collect();
    let input_im = vec![0.0f32; n];

    let (out_re, out_im) = run_fft(&device, &queue, &input_re, &input_im, false);

    // Magnitude spectrum
    let mag: Vec<f32> = out_re
        .iter()
        .zip(out_im.iter())
        .map(|(r, i)| (r * r + i * i).sqrt())
        .collect();

    // Peaks should be at bin 5 and bin 59 (= 64-5), each with magnitude N/2 = 32
    assert!(
        (mag[freq] - (n as f32 / 2.0)).abs() < 0.5,
        "Peak at bin {}: expected {}, got {}",
        freq,
        n as f32 / 2.0,
        mag[freq]
    );
    assert!(
        (mag[n - freq] - (n as f32 / 2.0)).abs() < 0.5,
        "Peak at bin {}: expected {}, got {}",
        n - freq,
        n as f32 / 2.0,
        mag[n - freq]
    );

    // Other bins should be near zero
    for i in 0..n {
        if i != freq && i != n - freq {
            assert!(
                mag[i] < 0.5,
                "Bin {} should be ~0, got {}",
                i, mag[i]
            );
        }
    }
    println!("✓ FFT cosine: peaks at ±{} Hz", freq);
}

#[test]
fn test_fft_roundtrip() {
    // FFT then IFFT should recover original signal
    let (device, queue) = create_test_device();
    let n = 32;
    let input_re: Vec<f32> = (0..n).map(|i| (i as f32 * 0.1).sin() + 0.5).collect();
    let input_im: Vec<f32> = (0..n).map(|i| (i as f32 * 0.2).cos()).collect();

    // Forward FFT
    let (fft_re, fft_im) = run_fft(&device, &queue, &input_re, &input_im, false);

    // Inverse FFT
    let (recovered_re, recovered_im) = run_fft(&device, &queue, &fft_re, &fft_im, true);

    assert_f32_near(&recovered_re, &input_re, 1e-3, "roundtrip real");
    assert_f32_near(&recovered_im, &input_im, 1e-3, "roundtrip imag");
    println!("✓ FFT roundtrip: forward→inverse recovers signal");
}

#[test]
fn test_fft_vs_cpu_dft() {
    // Compare GPU FFT against CPU DFT on random-ish data
    let (device, queue) = create_test_device();
    let n = 128;
    let input_re: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.618033988) % 1.0) - 0.5)
        .collect();
    let input_im: Vec<f32> = (0..n)
        .map(|i| ((i as f32 * 0.414213562) % 1.0) - 0.5)
        .collect();

    let (gpu_re, gpu_im) = run_fft(&device, &queue, &input_re, &input_im, false);
    let (cpu_re, cpu_im) = cpu_dft(&input_re, &input_im, false);

    assert_f32_near(&gpu_re, &cpu_re, 0.05, "GPU vs CPU DFT real");
    assert_f32_near(&gpu_im, &cpu_im, 0.05, "GPU vs CPU DFT imag");
    println!("✓ FFT matches CPU DFT (N={})", n);
}

#[test]
fn test_fft_multiple_rows() {
    // Test batch FFT on 4 rows simultaneously
    let (device, queue) = create_test_device();
    let n = 16u32;
    let num_rows = 4u32;
    let log2_n = (n as f32).log2() as u32;

    // Row 0: delta, Row 1: constant, Row 2: shifted delta, Row 3: zeros
    let mut input_re = vec![0.0f32; (n * num_rows) as usize];
    let input_im = vec![0.0f32; (n * num_rows) as usize];

    // Row 0: delta
    input_re[0] = 1.0;
    // Row 1: constant
    for i in 0..n as usize {
        input_re[n as usize + i] = 1.0;
    }
    // Row 2: delta at index 1
    input_re[2 * n as usize + 1] = 1.0;
    // Row 3: zeros (already)

    let params: [u32; 4] = [n, log2_n, num_rows, 0];
    let buf_size = (n * num_rows) as usize * 4;

    let result = run_compute_shader(
        &device,
        &queue,
        FFT_SHADER,
        "main",
        &[
            (0, u32_to_bytes(&params), BufferUsages::UNIFORM),
            (1, f32_to_bytes(&input_re), BufferUsages::STORAGE),
            (2, f32_to_bytes(&input_im), BufferUsages::STORAGE),
            (3, vec![0u8; buf_size], BufferUsages::STORAGE),
            (4, vec![0u8; buf_size], BufferUsages::STORAGE),
        ],
        3,
        (num_rows, 1, 1),
    );
    let out_re = bytes_to_f32(&result);

    // Row 0: FFT(delta) = all ones
    assert_f32_near(
        &out_re[0..n as usize],
        &vec![1.0; n as usize],
        1e-4,
        "Row 0: FFT(delta)",
    );

    // Row 1: FFT(constant) = [N, 0, ..., 0]
    assert!(
        (out_re[n as usize] - n as f32).abs() < 1e-3,
        "Row 1: DC = {}",
        out_re[n as usize]
    );
    for i in 1..n as usize {
        assert!(
            out_re[n as usize + i].abs() < 1e-3,
            "Row 1: bin {} = {}",
            i,
            out_re[n as usize + i]
        );
    }

    // Row 3: FFT(zeros) = all zeros
    assert_f32_near(
        &out_re[3 * n as usize..4 * n as usize],
        &vec![0.0; n as usize],
        1e-5,
        "Row 3: FFT(zeros)",
    );

    println!("✓ FFT batch: 4 rows processed correctly");
}

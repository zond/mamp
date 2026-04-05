//! Tests for 2D FFT (row FFT + column FFT using fft.wgsl)
//!
//! Run: cargo test --features native-test test_fft2d -- --nocapture

#[path = "gpu_test_harness.rs"]
mod gpu_test_harness;
use gpu_test_harness::*;

use std::f32::consts::PI;
use wgpu::BufferUsages;

/// Run a 2D FFT on the GPU: row FFT then column FFT (or inverse: col IFFT then row IFFT).
fn run_fft_2d(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    shader_src: &str,
    input_re: &[f32],
    input_im: &[f32],
    width: u32,
    height: u32,
    inverse: bool,
) -> (Vec<f32>, Vec<f32>) {
    let total = (width * height) as usize;
    let buf_bytes = total * 4;
    let log2_w = (width as f32).log2() as u32;
    let log2_h = (height as f32).log2() as u32;
    let inv = if inverse { 1u32 } else { 0 };

    // We need two passes. Since run_compute_shader creates fresh buffers each time,
    // we need to chain: pass 1 output → pass 2 input.

    if !inverse {
        // Forward: rows first, then columns

        // Pass 1: Row FFT — num_ffts=height, n=width, stride=1, fft_stride=width
        let row_params: [u32; 8] = [width, log2_w, height, inv, 1, width, 0, 0];
        let row_re = run_compute_shader(
            device, queue, shader_src, "main",
            &[
                (0, u32_to_bytes(&row_params), BufferUsages::UNIFORM),
                (1, f32_to_bytes(input_re), BufferUsages::STORAGE),
                (2, f32_to_bytes(input_im), BufferUsages::STORAGE),
                (3, vec![0u8; buf_bytes], BufferUsages::STORAGE),
                (4, vec![0u8; buf_bytes], BufferUsages::STORAGE),
            ],
            3, (height, 1, 1),
        );
        let row_im = run_compute_shader(
            device, queue, shader_src, "main",
            &[
                (0, u32_to_bytes(&row_params), BufferUsages::UNIFORM),
                (1, f32_to_bytes(input_re), BufferUsages::STORAGE),
                (2, f32_to_bytes(input_im), BufferUsages::STORAGE),
                (3, vec![0u8; buf_bytes], BufferUsages::STORAGE),
                (4, vec![0u8; buf_bytes], BufferUsages::STORAGE),
            ],
            4, (height, 1, 1),
        );

        // Pass 2: Column FFT — num_ffts=width, n=height, stride=width, fft_stride=1
        let col_params: [u32; 8] = [height, log2_h, width, inv, width, 1, 0, 0];
        let col_re = run_compute_shader(
            device, queue, shader_src, "main",
            &[
                (0, u32_to_bytes(&col_params), BufferUsages::UNIFORM),
                (1, row_re.clone(), BufferUsages::STORAGE),
                (2, row_im.clone(), BufferUsages::STORAGE),
                (3, vec![0u8; buf_bytes], BufferUsages::STORAGE),
                (4, vec![0u8; buf_bytes], BufferUsages::STORAGE),
            ],
            3, (width, 1, 1),
        );
        let col_im = run_compute_shader(
            device, queue, shader_src, "main",
            &[
                (0, u32_to_bytes(&col_params), BufferUsages::UNIFORM),
                (1, row_re, BufferUsages::STORAGE),
                (2, row_im, BufferUsages::STORAGE),
                (3, vec![0u8; buf_bytes], BufferUsages::STORAGE),
                (4, vec![0u8; buf_bytes], BufferUsages::STORAGE),
            ],
            4, (width, 1, 1),
        );

        (bytes_to_f32(&col_re), bytes_to_f32(&col_im))
    } else {
        // Inverse: columns first, then rows

        // Pass 1: Column IFFT
        let col_params: [u32; 8] = [height, log2_h, width, inv, width, 1, 0, 0];
        let col_re = run_compute_shader(
            device, queue, shader_src, "main",
            &[
                (0, u32_to_bytes(&col_params), BufferUsages::UNIFORM),
                (1, f32_to_bytes(input_re), BufferUsages::STORAGE),
                (2, f32_to_bytes(input_im), BufferUsages::STORAGE),
                (3, vec![0u8; buf_bytes], BufferUsages::STORAGE),
                (4, vec![0u8; buf_bytes], BufferUsages::STORAGE),
            ],
            3, (width, 1, 1),
        );
        let col_im = run_compute_shader(
            device, queue, shader_src, "main",
            &[
                (0, u32_to_bytes(&col_params), BufferUsages::UNIFORM),
                (1, f32_to_bytes(input_re), BufferUsages::STORAGE),
                (2, f32_to_bytes(input_im), BufferUsages::STORAGE),
                (3, vec![0u8; buf_bytes], BufferUsages::STORAGE),
                (4, vec![0u8; buf_bytes], BufferUsages::STORAGE),
            ],
            4, (width, 1, 1),
        );

        // Pass 2: Row IFFT
        let row_params: [u32; 8] = [width, log2_w, height, inv, 1, width, 0, 0];
        let row_re = run_compute_shader(
            device, queue, shader_src, "main",
            &[
                (0, u32_to_bytes(&row_params), BufferUsages::UNIFORM),
                (1, col_re.clone(), BufferUsages::STORAGE),
                (2, col_im.clone(), BufferUsages::STORAGE),
                (3, vec![0u8; buf_bytes], BufferUsages::STORAGE),
                (4, vec![0u8; buf_bytes], BufferUsages::STORAGE),
            ],
            3, (height, 1, 1),
        );
        let row_im = run_compute_shader(
            device, queue, shader_src, "main",
            &[
                (0, u32_to_bytes(&row_params), BufferUsages::UNIFORM),
                (1, col_re, BufferUsages::STORAGE),
                (2, col_im, BufferUsages::STORAGE),
                (3, vec![0u8; buf_bytes], BufferUsages::STORAGE),
                (4, vec![0u8; buf_bytes], BufferUsages::STORAGE),
            ],
            4, (height, 1, 1),
        );

        (bytes_to_f32(&row_re), bytes_to_f32(&row_im))
    }
}

/// CPU 2D DFT (separable: row DFT then column DFT).
fn cpu_dft_2d(
    re: &[f32], im: &[f32], width: usize, height: usize, inverse: bool,
) -> (Vec<f32>, Vec<f32>) {
    let sign = if inverse { 1.0 } else { -1.0 };
    let total = width * height;

    // Row-wise DFT
    let mut row_re = vec![0.0f32; total];
    let mut row_im = vec![0.0f32; total];
    let row_scale = if inverse { 1.0 / width as f32 } else { 1.0 };
    for y in 0..height {
        for kx in 0..width {
            let (mut sr, mut si) = (0.0, 0.0);
            for nx in 0..width {
                let angle = sign * 2.0 * PI * kx as f32 * nx as f32 / width as f32;
                let (c, s) = (angle.cos(), angle.sin());
                let idx = y * width + nx;
                sr += re[idx] * c - im[idx] * s;
                si += re[idx] * s + im[idx] * c;
            }
            row_re[y * width + kx] = sr * row_scale;
            row_im[y * width + kx] = si * row_scale;
        }
    }

    // Column-wise DFT
    let mut out_re = vec![0.0f32; total];
    let mut out_im = vec![0.0f32; total];
    let col_scale = if inverse { 1.0 / height as f32 } else { 1.0 };
    for x in 0..width {
        for ky in 0..height {
            let (mut sr, mut si) = (0.0, 0.0);
            for ny in 0..height {
                let angle = sign * 2.0 * PI * ky as f32 * ny as f32 / height as f32;
                let (c, s) = (angle.cos(), angle.sin());
                let idx = ny * width + x;
                sr += row_re[idx] * c - row_im[idx] * s;
                si += row_re[idx] * s + row_im[idx] * c;
            }
            out_re[ky * width + x] = sr * col_scale;
            out_im[ky * width + x] = si * col_scale;
        }
    }

    (out_re, out_im)
}

#[test]
fn test_2d_fft_delta() {
    // 2D delta at (0,0) → flat spectrum (all ones)
    let (device, queue) = create_test_device();
    let src = fft_shader_source(1024);
    let w = 8u32;
    let h = 8u32;
    let total = (w * h) as usize;

    let mut re = vec![0.0f32; total];
    re[0] = 1.0;
    let im = vec![0.0f32; total];

    let (out_re, out_im) = run_fft_2d(&device, &queue, &src, &re, &im, w, h, false);

    assert_f32_near(&out_re, &vec![1.0; total], 1e-3, "2D delta re");
    assert_f32_near(&out_im, &vec![0.0; total], 1e-3, "2D delta im");
    println!("✓ 2D FFT delta {}x{}: flat spectrum", w, h);
}

#[test]
fn test_2d_fft_constant() {
    // 2D constant → DC peak at (0,0) = W*H
    let (device, queue) = create_test_device();
    let src = fft_shader_source(1024);
    let w = 16u32;
    let h = 8u32;
    let total = (w * h) as usize;

    let re = vec![1.0f32; total];
    let im = vec![0.0f32; total];

    let (out_re, _out_im) = run_fft_2d(&device, &queue, &src, &re, &im, w, h, false);

    assert!((out_re[0] - (w * h) as f32).abs() < 0.1,
        "DC should be {}, got {}", w * h, out_re[0]);
    for i in 1..total {
        assert!(out_re[i].abs() < 0.1, "bin {} should be ~0, got {}", i, out_re[i]);
    }
    println!("✓ 2D FFT constant {}x{}: DC = {}", w, h, out_re[0]);
}

#[test]
fn test_2d_fft_roundtrip() {
    // Forward 2D FFT then inverse should recover original
    let (device, queue) = create_test_device();
    let src = fft_shader_source(1024);
    let w = 16u32;
    let h = 16u32;
    let total = (w * h) as usize;

    let re: Vec<f32> = (0..total).map(|i| ((i as f32 * 0.37) % 1.0) - 0.5).collect();
    let im: Vec<f32> = (0..total).map(|i| ((i as f32 * 0.73) % 1.0) - 0.5).collect();

    let (fft_re, fft_im) = run_fft_2d(&device, &queue, &src, &re, &im, w, h, false);
    let (rec_re, rec_im) = run_fft_2d(&device, &queue, &src, &fft_re, &fft_im, w, h, true);

    assert_f32_near(&rec_re, &re, 0.01, "2D roundtrip re");
    assert_f32_near(&rec_im, &im, 0.01, "2D roundtrip im");
    println!("✓ 2D FFT roundtrip {}x{}", w, h);
}

#[test]
fn test_2d_fft_vs_cpu() {
    // Compare GPU 2D FFT against CPU 2D DFT
    let (device, queue) = create_test_device();
    let src = fft_shader_source(1024);
    let w = 16u32;
    let h = 16u32;
    let total = (w * h) as usize;

    let re: Vec<f32> = (0..total).map(|i| ((i as f32 * 0.618) % 1.0) - 0.5).collect();
    let im = vec![0.0f32; total];

    let (gpu_re, gpu_im) = run_fft_2d(&device, &queue, &src, &re, &im, w, h, false);
    let (cpu_re, cpu_im) = cpu_dft_2d(&re, &im, w as usize, h as usize, false);

    assert_f32_near(&gpu_re, &cpu_re, 0.1, "2D GPU vs CPU re");
    assert_f32_near(&gpu_im, &cpu_im, 0.1, "2D GPU vs CPU im");
    println!("✓ 2D FFT matches CPU DFT {}x{}", w, h);
}

#[test]
fn test_2d_fft_2d_cosine() {
    // 2D cosine cos(2π·fx·x/W) · cos(2π·fy·y/H)
    // Should have peaks at (±fx, ±fy)
    let (device, queue) = create_test_device();
    let src = fft_shader_source(1024);
    let w = 32u32;
    let h = 32u32;
    let total = (w * h) as usize;
    let fx = 3usize;
    let fy = 5usize;

    let re: Vec<f32> = (0..total).map(|i| {
        let x = (i % w as usize) as f32;
        let y = (i / w as usize) as f32;
        (2.0 * PI * fx as f32 * x / w as f32).cos()
        * (2.0 * PI * fy as f32 * y / h as f32).cos()
    }).collect();
    let im = vec![0.0f32; total];

    let (out_re, out_im) = run_fft_2d(&device, &queue, &src, &re, &im, w, h, false);

    let mag: Vec<f32> = out_re.iter().zip(out_im.iter())
        .map(|(r, i)| (r * r + i * i).sqrt()).collect();

    // cos(a)cos(b) = 0.5[cos(a+b) + cos(a-b)]
    // Peaks at (fx,fy), (W-fx,fy), (fx,H-fy), (W-fx,H-fy)
    let expected = (w * h) as f32 / 4.0;
    let peaks = [
        (fy, fx),
        (fy, w as usize - fx),
        (h as usize - fy, fx),
        (h as usize - fy, w as usize - fx),
    ];
    for &(py, px) in &peaks {
        let idx = py * w as usize + px;
        assert!((mag[idx] - expected).abs() < 2.0,
            "Peak at ({},{}) idx={}: expected {}, got {}", px, py, idx, expected, mag[idx]);
    }

    println!("✓ 2D FFT cosine {}x{}: peaks at fx={}, fy={}", w, h, fx, fy);
}

#[test]
fn test_2d_fft_rectangular() {
    // Non-square: 32x16 roundtrip
    let (device, queue) = create_test_device();
    let src = fft_shader_source(1024);
    let w = 32u32;
    let h = 16u32;
    let total = (w * h) as usize;

    let re: Vec<f32> = (0..total).map(|i| (i as f32 * 0.123).sin()).collect();
    let im = vec![0.0f32; total];

    let (fft_re, fft_im) = run_fft_2d(&device, &queue, &src, &re, &im, w, h, false);
    let (rec_re, _) = run_fft_2d(&device, &queue, &src, &fft_re, &fft_im, w, h, true);

    assert_f32_near(&rec_re, &re, 0.02, "rectangular roundtrip");
    println!("✓ 2D FFT rectangular {}x{} roundtrip", w, h);
}

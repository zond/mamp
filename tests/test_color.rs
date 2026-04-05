//! Tests for rgba_to_y.wgsl and yiq_to_rgba.wgsl color conversion
//!
//! Run: cargo test --features native-test --test test_color -- --nocapture

#[path = "gpu_test_harness.rs"]
mod gpu_test_harness;
use gpu_test_harness::*;

use wgpu::BufferUsages;

const RGBA_TO_Y: &str = include_str!("../src/shaders/rgba_to_y.wgsl");
const YIQ_TO_RGBA: &str = include_str!("../src/shaders/yiq_to_rgba.wgsl");

#[test]
fn test_yiq_roundtrip() {
    // RGBA → Y,I,Q → RGBA should recover original (within quantization)
    let (device, queue) = create_test_device();
    let w = 8u32;
    let h = 8u32;
    let pw = 8u32; // no padding for this test
    let ph = 8u32;
    let total = (w * h) as usize;
    let ptotal = (pw * ph) as usize;

    // Create test image: gradient
    let rgba: Vec<u32> = (0..total).map(|i| {
        let r = ((i * 17) % 256) as u32;
        let g = ((i * 31) % 256) as u32;
        let b = ((i * 53) % 256) as u32;
        r | (g << 8) | (b << 16) | (255 << 24)
    }).collect();

    // RGBA → Y, I, Q
    let params_fwd: [u32; 4] = [w, h, pw, ph];
    let s = BufferUsages::STORAGE;
    let pbytes = ptotal * 4;

    let y_buf = run_compute_shader(&device, &queue, RGBA_TO_Y, "main", &[
        (0, u32_to_bytes(&params_fwd), BufferUsages::UNIFORM),
        (1, u32_to_bytes(&rgba), s),
        (2, vec![0u8; pbytes], s),
        (3, vec![0u8; pbytes], s),
        (4, vec![0u8; pbytes], s),
    ], 2, (pw.div_ceil(16), ph.div_ceil(16), 1));
    let i_buf = run_compute_shader(&device, &queue, RGBA_TO_Y, "main", &[
        (0, u32_to_bytes(&params_fwd), BufferUsages::UNIFORM),
        (1, u32_to_bytes(&rgba), s),
        (2, vec![0u8; pbytes], s),
        (3, vec![0u8; pbytes], s),
        (4, vec![0u8; pbytes], s),
    ], 3, (pw.div_ceil(16), ph.div_ceil(16), 1));
    let q_buf = run_compute_shader(&device, &queue, RGBA_TO_Y, "main", &[
        (0, u32_to_bytes(&params_fwd), BufferUsages::UNIFORM),
        (1, u32_to_bytes(&rgba), s),
        (2, vec![0u8; pbytes], s),
        (3, vec![0u8; pbytes], s),
        (4, vec![0u8; pbytes], s),
    ], 4, (pw.div_ceil(16), ph.div_ceil(16), 1));

    // Y, I, Q → RGBA
    let params_inv: [u32; 4] = [w, h, pw, ph];
    let result = run_compute_shader(&device, &queue, YIQ_TO_RGBA, "main", &[
        (0, u32_to_bytes(&params_inv), BufferUsages::UNIFORM),
        (1, y_buf, s),
        (2, i_buf, s),
        (3, q_buf, s),
        (4, vec![0u8; total * 4], s),
    ], 4, (w.div_ceil(16), h.div_ceil(16), 1));
    let output = bytes_to_u32(&result);

    // Compare: each channel should match within ±2 (quantization)
    let mut max_err = 0u32;
    for i in 0..total {
        let orig = rgba[i];
        let out = output[i];
        for shift in [0u32, 8, 16] {
            let o = (orig >> shift) & 0xFF;
            let r = (out >> shift) & 0xFF;
            let err = if o > r { o - r } else { r - o };
            if err > max_err { max_err = err; }
        }
    }

    println!("  YIQ roundtrip max channel error: {}", max_err);
    assert!(max_err <= 2, "Max error {} exceeds tolerance", max_err);
    println!("✓ YIQ roundtrip: max error {} (within quantization)", max_err);
}

#[test]
fn test_padding() {
    // Verify zero-padding works: 4×4 image padded to 8×8
    let (device, queue) = create_test_device();
    let w = 4u32;
    let h = 4u32;
    let pw = 8u32;
    let ph = 8u32;
    let total = (w * h) as usize;
    let ptotal = (pw * ph) as usize;

    let rgba: Vec<u32> = vec![0x00FFFFFFu32; total]; // white
    let params: [u32; 4] = [w, h, pw, ph];
    let s = BufferUsages::STORAGE;

    let y_buf = run_compute_shader(&device, &queue, RGBA_TO_Y, "main", &[
        (0, u32_to_bytes(&params), BufferUsages::UNIFORM),
        (1, u32_to_bytes(&rgba), s),
        (2, vec![0u8; ptotal * 4], s),
        (3, vec![0u8; ptotal * 4], s),
        (4, vec![0u8; ptotal * 4], s),
    ], 2, (pw.div_ceil(16), ph.div_ceil(16), 1));
    let y = bytes_to_f32(&y_buf);

    // First 4×4 should be ~1.0 (white luminance)
    for row in 0..h as usize {
        for col in 0..w as usize {
            let val = y[row * pw as usize + col];
            assert!((val - 1.0).abs() < 0.01,
                "({},{}) should be ~1.0, got {}", col, row, val);
        }
    }

    // Padded region should be 0
    for row in 0..ph as usize {
        for col in 0..pw as usize {
            if row >= h as usize || col >= w as usize {
                let val = y[row * pw as usize + col];
                assert!(val.abs() < 0.01,
                    "Padded ({},{}) should be 0, got {}", col, row, val);
            }
        }
    }

    println!("✓ Padding: 4×4→8×8, image=1.0, padding=0.0");
}

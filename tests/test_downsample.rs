//! Test: gaussian_downsample.wgsl
//! Verifies 2x2 average downsampling produces correct results.
//!
//! Run: cargo test --features native-test test_downsample

#[path = "gpu_test_harness.rs"]
mod gpu_test_harness;
use gpu_test_harness::*;

use wgpu::BufferUsages;

const SHADER: &str = include_str!("../src/shaders/gaussian_downsample.wgsl");

#[test]
fn test_downsample_4x4_to_2x2() {
    let (device, queue) = create_test_device();

    // 4x4 input, 3 channels (CHW layout)
    // Channel 0 (R): all 1.0
    // Channel 1 (G): gradient 0..15 / 15
    // Channel 2 (B): all 0.5
    let in_w: u32 = 4;
    let in_h: u32 = 4;
    let out_w: u32 = 2;
    let out_h: u32 = 2;
    let in_hw = (in_w * in_h) as usize;
    let out_hw = (out_w * out_h) as usize;

    let mut input = vec![0.0f32; 3 * in_hw];
    // Channel 0: all 1.0
    for i in 0..in_hw {
        input[i] = 1.0;
    }
    // Channel 1: gradient
    for i in 0..in_hw {
        input[in_hw + i] = i as f32 / (in_hw - 1) as f32;
    }
    // Channel 2: all 0.5
    for i in 0..in_hw {
        input[2 * in_hw + i] = 0.5;
    }

    // Params: in_width, in_height, out_width, out_height (4 x u32)
    let params: [u32; 4] = [in_w, in_h, out_w, out_h];

    let result_bytes = run_compute_shader(
        &device,
        &queue,
        SHADER,
        "main",
        &[
            (0, u32_to_bytes(&params), BufferUsages::UNIFORM),
            (1, f32_to_bytes(&input), BufferUsages::STORAGE),
            (2, vec![0u8; 3 * out_hw * 4], BufferUsages::STORAGE),
        ],
        2, // output binding
        (out_w.div_ceil(16), out_h.div_ceil(16), 1),
    );

    let output = bytes_to_f32(&result_bytes);

    // Channel 0: average of four 1.0s = 1.0
    for i in 0..out_hw {
        assert!(
            (output[i] - 1.0).abs() < 1e-5,
            "Channel 0, pixel {}: expected 1.0, got {}",
            i,
            output[i]
        );
    }

    // Channel 2: average of four 0.5s = 0.5
    for i in 0..out_hw {
        assert!(
            (output[2 * out_hw + i] - 0.5).abs() < 1e-5,
            "Channel 2, pixel {}: expected 0.5, got {}",
            i,
            output[2 * out_hw + i]
        );
    }

    // Channel 1: each output pixel is avg of 2x2 block from gradient
    // pixel (0,0) = avg of input (0,0),(1,0),(0,1),(1,1) = avg(0/15, 1/15, 4/15, 5/15) = 10/60 = 1/6
    let expected_ch1 = [
        (0.0 + 1.0 + 4.0 + 5.0) / 4.0 / 15.0,   // (0,0)
        (2.0 + 3.0 + 6.0 + 7.0) / 4.0 / 15.0,   // (1,0)
        (8.0 + 9.0 + 12.0 + 13.0) / 4.0 / 15.0,  // (0,1)
        (10.0 + 11.0 + 14.0 + 15.0) / 4.0 / 15.0, // (1,1)
    ];
    assert_f32_near(
        &output[out_hw..2 * out_hw],
        &expected_ch1,
        1e-4,
        "Channel 1 (gradient downsample)",
    );

    println!("✓ gaussian_downsample: 4x4→2x2 correct");
}

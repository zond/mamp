//! Shared GPU test harness for shader unit tests.
//!
//! Run with: cargo test --features native-test
//!
//! Each test creates a real GPU device (Vulkan), uploads known input,
//! runs a compute shader, reads back the output, and asserts correctness.

use bytemuck;
use wgpu::*;

/// Create a native GPU device for testing.
pub fn create_test_device() -> (Device, Queue) {
    let mut desc = InstanceDescriptor::new_without_display_handle();
    desc.backends = Backends::VULKAN;
    let instance = Instance::new(desc);

    let adapter = pollster::block_on(instance.request_adapter(&RequestAdapterOptions {
        power_preference: PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("No Vulkan adapter found — is a GPU available?");

    println!("Test GPU: {:?}", adapter.get_info().name);

    pollster::block_on(adapter.request_device(&DeviceDescriptor {
        label: Some("test-device"),
        required_features: Features::empty(),
        required_limits: Limits::default(),
        memory_hints: MemoryHints::default(),
        trace: wgpu::Trace::Off,
        experimental_features: Default::default(),
    }))
    .expect("Failed to create test device")
}

/// Run a compute shader and read back results.
///
/// - `device`/`queue`: from `create_test_device()`
/// - `shader_src`: WGSL source code
/// - `entry_point`: entry function name
/// - `bind_group_entries`: list of (binding, data_slice, usage) for each buffer
/// - `output_binding`: which binding index is the output
/// - `output_size`: expected output size in bytes
/// - `workgroups`: (x, y, z) dispatch dimensions
///
/// Returns the output buffer contents as bytes.
pub fn run_compute_shader(
    device: &Device,
    queue: &Queue,
    shader_src: &str,
    entry_point: &str,
    buffers: &[(u32, Vec<u8>, BufferUsages)], // (binding, initial_data, usage)
    output_binding: u32,
    workgroups: (u32, u32, u32),
) -> Vec<u8> {
    let module = device.create_shader_module(ShaderModuleDescriptor {
        label: Some("test-shader"),
        source: ShaderSource::Wgsl(shader_src.into()),
    });

    let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("test-pipeline"),
        layout: None,
        module: &module,
        entry_point: Some(entry_point),
        compilation_options: Default::default(),
        cache: None,
    });

    // Create buffers
    let gpu_buffers: Vec<(u32, Buffer)> = buffers
        .iter()
        .map(|(binding, data, usage)| {
            let buf = device.create_buffer(&BufferDescriptor {
                label: Some(&format!("buf-{}", binding)),
                size: data.len() as u64,
                usage: *usage | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, data);
            (*binding, buf)
        })
        .collect();

    // Create bind group
    let entries: Vec<BindGroupEntry> = gpu_buffers
        .iter()
        .map(|(binding, buf)| BindGroupEntry {
            binding: *binding,
            resource: buf.as_entire_binding(),
        })
        .collect();

    let bind_group = device.create_bind_group(&BindGroupDescriptor {
        label: Some("test-bg"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &entries,
    });

    // Dispatch
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("test-encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("test-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, Some(&bind_group), &[]);
        pass.dispatch_workgroups(workgroups.0, workgroups.1, workgroups.2);
    }

    // Find output buffer and copy to staging
    let output_buf = gpu_buffers
        .iter()
        .find(|(b, _)| *b == output_binding)
        .expect("Output binding not found")
        .1
        .size();

    let staging = device.create_buffer(&BufferDescriptor {
        label: Some("staging"),
        size: output_buf,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let out_gpu_buf = &gpu_buffers
        .iter()
        .find(|(b, _)| *b == output_binding)
        .unwrap()
        .1;
    encoder.copy_buffer_to_buffer(out_gpu_buf, 0, &staging, 0, output_buf);
    queue.submit(std::iter::once(encoder.finish()));

    // Read back
    let slice = staging.slice(..);
    slice.map_async(MapMode::Read, |_| {});
    device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();

    let data = slice.get_mapped_range().to_vec();
    staging.unmap();
    data
}

/// Helper: convert f32 slice to bytes for buffer upload.
pub fn f32_to_bytes(data: &[f32]) -> Vec<u8> {
    bytemuck::cast_slice(data).to_vec()
}

/// Helper: convert u32 slice to bytes for buffer upload.
pub fn u32_to_bytes(data: &[u32]) -> Vec<u8> {
    bytemuck::cast_slice(data).to_vec()
}

/// Helper: convert bytes back to f32 slice.
pub fn bytes_to_f32(data: &[u8]) -> Vec<f32> {
    bytemuck::cast_slice(data).to_vec()
}

/// Helper: convert bytes back to u32 slice.
pub fn bytes_to_u32(data: &[u8]) -> Vec<u32> {
    bytemuck::cast_slice(data).to_vec()
}

/// Assert two f32 slices are approximately equal.
pub fn assert_f32_near(actual: &[f32], expected: &[f32], tolerance: f32, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{}: length mismatch: {} vs {}",
        context,
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tolerance,
            "{}: index {}: actual={}, expected={}, diff={}",
            context,
            i,
            a,
            e,
            (a - e).abs()
        );
    }
}

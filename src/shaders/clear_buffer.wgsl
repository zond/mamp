// clear_buffer.wgsl — Zero-fill two buffers (replaces copy_buffer_to_buffer)

struct Dims { width: u32, height: u32 }

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read_write> buf_a: array<f32>;
@group(0) @binding(2) var<storage, read_write> buf_b: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= dims.width || gid.y >= dims.height { return; }
    let idx = gid.y * dims.width + gid.x;
    buf_a[idx] = 0.0;
    buf_b[idx] = 0.0;
}

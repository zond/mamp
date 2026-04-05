// add_buffers.wgsl — Element-wise addition: out[i] = a[i] + b[i]

struct Dims { width: u32, height: u32 }

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= dims.width || gid.y >= dims.height { return; }
    let idx = gid.y * dims.width + gid.x;
    out[idx] = a[idx] + b[idx];
}

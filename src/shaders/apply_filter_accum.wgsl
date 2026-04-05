// apply_filter_accum.wgsl — Multiply spectrum by pre-computed filter and accumulate
//
// accum += in * filter_buf[idx]

struct Dims {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> filter_buf: array<f32>;
@group(0) @binding(2) var<storage, read> input_re: array<f32>;
@group(0) @binding(3) var<storage, read> input_im: array<f32>;
@group(0) @binding(4) var<storage, read_write> accum_re: array<f32>;
@group(0) @binding(5) var<storage, read_write> accum_im: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= dims.width || y >= dims.height { return; }

    let idx = y * dims.width + x;
    let h = filter_buf[idx];
    accum_re[idx] += input_re[idx] * h;
    accum_im[idx] += input_im[idx] * h;
}

// apply_filter.wgsl — Multiply spectrum by pre-computed filter (1 read, no trig)
//
// out = in * filter_buf[idx]

struct Dims {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> filter_buf: array<f32>;
@group(0) @binding(2) var<storage, read> input_re: array<f32>;
@group(0) @binding(3) var<storage, read> input_im: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_re: array<f32>;
@group(0) @binding(5) var<storage, read_write> output_im: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= dims.width || y >= dims.height { return; }

    let idx = y * dims.width + x;
    let h = filter_buf[idx];
    output_re[idx] = input_re[idx] * h;
    output_im[idx] = input_im[idx] * h;
}

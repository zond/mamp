// rgba_to_chw.wgsl — Convert packed RGBA u32 pixels to CHW float tensor

struct FrameParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: FrameParams;
@group(0) @binding(1) var<storage, read> rgba_input: array<u32>;
@group(0) @binding(2) var<storage, read_write> chw_output: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if x >= params.width || y >= params.height {
        return;
    }

    let pixel_idx = y * params.width + x;
    let packed = rgba_input[pixel_idx];

    let r = f32(packed & 0xFFu) / 255.0;
    let g = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((packed >> 16u) & 0xFFu) / 255.0;

    let hw = params.height * params.width;
    chw_output[0u * hw + pixel_idx] = r;
    chw_output[1u * hw + pixel_idx] = g;
    chw_output[2u * hw + pixel_idx] = b;
}

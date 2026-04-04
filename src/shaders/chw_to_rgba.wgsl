// chw_to_rgba.wgsl — Convert CHW float tensor to packed RGBA u32 pixels

struct FrameParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: FrameParams;
@group(0) @binding(1) var<storage, read> chw_input: array<f32>;
@group(0) @binding(2) var<storage, read_write> rgba_output: array<u32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;

    if x >= params.width || y >= params.height {
        return;
    }

    let pixel_idx = y * params.width + x;
    let hw = params.height * params.width;

    let r = u32(clamp(chw_input[0u * hw + pixel_idx] * 255.0, 0.0, 255.0));
    let g = u32(clamp(chw_input[1u * hw + pixel_idx] * 255.0, 0.0, 255.0));
    let b = u32(clamp(chw_input[2u * hw + pixel_idx] * 255.0, 0.0, 255.0));

    rgba_output[pixel_idx] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}

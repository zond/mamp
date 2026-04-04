// chw_to_rgba.wgsl — Convert CHW float tensor to packed RGBA u32 pixels (optimized)
//
// OPTIMIZATION: Each thread processes 4 horizontally consecutive pixels instead of 1.
// This reduces total thread count by 4x and amortizes bounds checking. Three channel
// planes are gathered, clamped, packed to u32, and stored.
// Workgroup size increased to 16x16 since this shader is very lightweight per thread.
//
// DISPATCH CHANGE: x-dimension now covers ceil(W/4) pixels, divided by workgroup width 16.
//   Old: dispatch(ceil(W/8),   ceil(H/8),  1)  with workgroup_size(8,8,1)
//   New: dispatch(ceil(W/64),  ceil(H/16), 1)  with workgroup_size(16,16,1)
//   More precisely: dispatch(ceil(ceil(W/4)/16), ceil(H/16), 1)
//   Or equivalently: dispatch(ceil(W/64), ceil(H/16), 1)

struct FrameParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: FrameParams;
@group(0) @binding(1) var<storage, read> chw_input: array<f32>;
@group(0) @binding(2) var<storage, read_write> rgba_output: array<u32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    // Each thread handles 4 consecutive pixels in x
    let x_base = gid.x * 4u;
    let y = gid.y;

    if y >= params.height || x_base >= params.width {
        return;
    }

    let hw = params.height * params.width;
    let row_offset = y * params.width;

    // Determine how many pixels to process (1-4, handling right edge)
    let x_end = min(x_base + 4u, params.width);

    for (var x = x_base; x < x_end; x = x + 1u) {
        let pixel_idx = row_offset + x;

        let r = u32(clamp(chw_input[pixel_idx] * 255.0, 0.0, 255.0));
        let g = u32(clamp(chw_input[hw + pixel_idx] * 255.0, 0.0, 255.0));
        let b = u32(clamp(chw_input[2u * hw + pixel_idx] * 255.0, 0.0, 255.0));

        rgba_output[pixel_idx] = r | (g << 8u) | (b << 16u) | (255u << 24u);
    }
}

// rgba_to_chw.wgsl — Convert packed RGBA u32 pixels to CHW float tensor (optimized)
//
// OPTIMIZATION: Each thread processes 4 horizontally consecutive pixels instead of 1.
// This reduces total thread count by 4x and amortizes bounds checking. The 4 packed u32
// values are loaded, unpacked to RGB floats, and scattered to 3 channel planes.
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
@group(0) @binding(1) var<storage, read> rgba_input: array<u32>;
@group(0) @binding(2) var<storage, read_write> chw_output: array<f32>;

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
        let packed = rgba_input[pixel_idx];

        let r = f32(packed & 0xFFu) / 255.0;
        let g = f32((packed >> 8u) & 0xFFu) / 255.0;
        let b = f32((packed >> 16u) & 0xFFu) / 255.0;

        chw_output[pixel_idx]          = r;
        chw_output[hw + pixel_idx]     = g;
        chw_output[2u * hw + pixel_idx] = b;
    }
}

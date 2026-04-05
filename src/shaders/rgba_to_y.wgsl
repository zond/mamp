// rgba_to_y.wgsl — Extract luminance from packed RGBA u32, pad to FFT dimensions
//
// Input:  RGBA u32 buffer at (orig_width × orig_height)
// Output: Y float buffer at (padded_width × padded_height), zero-padded
//
// Also stores I, Q chrominance for later recombination.

struct Params {
    orig_width: u32,
    orig_height: u32,
    padded_width: u32,
    padded_height: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> rgba_input: array<u32>;
@group(0) @binding(2) var<storage, read_write> y_output: array<f32>;
@group(0) @binding(3) var<storage, read_write> i_output: array<f32>;
@group(0) @binding(4) var<storage, read_write> q_output: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.padded_width || y >= params.padded_height { return; }

    let out_idx = y * params.padded_width + x;

    // Outside original image: mirror/reflect padding (reduces FFT edge ringing)
    var sx = x;
    var sy = y;
    if sx >= params.orig_width {
        sx = 2u * params.orig_width - sx - 2u;
    }
    if sy >= params.orig_height {
        sy = 2u * params.orig_height - sy - 2u;
    }
    sx = clamp(sx, 0u, params.orig_width - 1u);
    sy = clamp(sy, 0u, params.orig_height - 1u);

    let in_idx = sy * params.orig_width + sx;
    let packed = rgba_input[in_idx];

    let r = f32(packed & 0xFFu) / 255.0;
    let g = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((packed >> 16u) & 0xFFu) / 255.0;

    // YIQ color space
    y_output[out_idx] = 0.299 * r + 0.587 * g + 0.114 * b;
    i_output[out_idx] = 0.596 * r - 0.275 * g - 0.321 * b;
    q_output[out_idx] = 0.212 * r - 0.523 * g + 0.311 * b;
}

// yiq_to_rgba.wgsl — Recombine modified Y with original I,Q → packed RGBA u32
//
// Crops from padded dimensions back to original dimensions.

struct Params {
    orig_width: u32,
    orig_height: u32,
    padded_width: u32,
    padded_height: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> y_input: array<f32>;   // modified luminance
@group(0) @binding(2) var<storage, read> i_input: array<f32>;   // original chrominance I
@group(0) @binding(3) var<storage, read> q_input: array<f32>;   // original chrominance Q
@group(0) @binding(4) var<storage, read_write> rgba_output: array<u32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.orig_width || y >= params.orig_height { return; }

    let padded_idx = y * params.padded_width + x;
    let out_idx = y * params.orig_width + x;

    let luma = y_input[padded_idx];
    let i_val = i_input[padded_idx];
    let q_val = q_input[padded_idx];

    // YIQ → RGB
    let r = clamp(luma + 0.956 * i_val + 0.621 * q_val, 0.0, 1.0);
    let g = clamp(luma - 0.272 * i_val - 0.647 * q_val, 0.0, 1.0);
    let b = clamp(luma - 1.107 * i_val + 1.704 * q_val, 0.0, 1.0);

    let ri = u32(r * 255.0);
    let gi = u32(g * 255.0);
    let bi = u32(b * 255.0);

    rgba_output[out_idx] = ri | (gi << 8u) | (bi << 16u) | (255u << 24u);
}

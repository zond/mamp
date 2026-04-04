// laplacian_temporal.wgsl — Compute Laplacian + IIR temporal bandpass + amplify
//
// For normal levels: laplacian = gaussian[i] - bilinear_upsample(gaussian[i+1])
// For coarsest level: output = gaussian[i] (residual, no filtering)
// Temporal IIR bandpass isolates motion at the target frequency band.
// Output = laplacian + amplification * bandpass

struct Params {
    width: u32,
    height: u32,
    coarse_width: u32,
    coarse_height: u32,
    alpha_low: f32,
    alpha_high: f32,
    amplification: f32,
    is_coarsest: u32,  // 1 = no coarser level, skip subtraction + filtering
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gaussian_fine: array<f32>;   // G[i]
@group(0) @binding(2) var<storage, read> gaussian_coarse: array<f32>; // G[i+1] (unused if coarsest)
@group(0) @binding(3) var<storage, read_write> lp_high: array<f32>;   // IIR state
@group(0) @binding(4) var<storage, read_write> lp_low: array<f32>;    // IIR state
@group(0) @binding(5) var<storage, read_write> output: array<f32>;    // amplified Laplacian

// Bilinear sample from coarse level
fn sample_coarse(c: u32, y: f32, x: f32) -> f32 {
    let cw = params.coarse_width;
    let ch = params.coarse_height;
    let x0 = u32(max(floor(x), 0.0));
    let y0 = u32(max(floor(y), 0.0));
    let x1 = min(x0 + 1u, cw - 1u);
    let y1 = min(y0 + 1u, ch - 1u);
    let fx = x - floor(x);
    let fy = y - floor(y);
    let base = c * ch * cw;
    let v00 = gaussian_coarse[base + y0 * cw + x0];
    let v10 = gaussian_coarse[base + y0 * cw + x1];
    let v01 = gaussian_coarse[base + y1 * cw + x0];
    let v11 = gaussian_coarse[base + y1 * cw + x1];
    return v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy) +
           v01 * (1.0 - fx) * fy + v11 * fx * fy;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    if ox >= params.width || oy >= params.height { return; }

    let hw = params.height * params.width;
    let idx = oy * params.width + ox;

    if params.is_coarsest == 1u {
        // Residual: just copy, no temporal filtering
        for (var c = 0u; c < 3u; c++) {
            output[c * hw + idx] = gaussian_fine[c * hw + idx];
        }
        return;
    }

    // Map fine pixel to coarse coordinates for bilinear upsample
    let scale_x = f32(params.coarse_width) / f32(params.width);
    let scale_y = f32(params.coarse_height) / f32(params.height);
    let src_x = (f32(ox) + 0.5) * scale_x - 0.5;
    let src_y = (f32(oy) + 0.5) * scale_y - 0.5;

    for (var c = 0u; c < 3u; c++) {
        let i = c * hw + idx;
        let fine_val = gaussian_fine[i];
        let coarse_val = sample_coarse(c, src_y, src_x);

        // Laplacian = fine - upsampled coarse
        let laplacian = fine_val - coarse_val;

        // IIR temporal bandpass
        let new_lp_high = params.alpha_high * laplacian + (1.0 - params.alpha_high) * lp_high[i];
        let new_lp_low = params.alpha_low * laplacian + (1.0 - params.alpha_low) * lp_low[i];
        lp_high[i] = new_lp_high;
        lp_low[i] = new_lp_low;
        let bandpass = new_lp_high - new_lp_low;

        // Amplified Laplacian = original + magnified temporal signal
        output[i] = laplacian + params.amplification * bandpass;
    }
}

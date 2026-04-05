// phase_amplify.wgsl — Phase-based motion amplification per sub-band
//
// For each pixel:
// 1. Compute phase difference between current and previous frame
//    using conjugate product (avoids explicit phase unwrapping)
// 2. IIR temporal bandpass filter the phase difference
// 3. Amplify the filtered phase
// 4. Rotate the current sub-band by the amplified phase
// 5. Store current as previous for next frame
//
// Dispatch: (ceil(width/16), ceil(height/16), 1)

struct Params {
    width: u32,
    height: u32,
    amplification: f32,
    alpha_low: f32,     // IIR coefficient for low cutoff
    alpha_high: f32,    // IIR coefficient for high cutoff
    is_first_frame: u32, // 1 = initialize state, don't amplify
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> cur_re: array<f32>;
@group(0) @binding(2) var<storage, read> cur_im: array<f32>;
@group(0) @binding(3) var<storage, read_write> prev_re: array<f32>;
@group(0) @binding(4) var<storage, read_write> prev_im: array<f32>;
@group(0) @binding(5) var<storage, read_write> lp_high: array<f32>;
@group(0) @binding(6) var<storage, read_write> lp_low: array<f32>;
@group(0) @binding(7) var<storage, read_write> out_re: array<f32>;
@group(0) @binding(8) var<storage, read_write> out_im: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height { return; }

    let idx = y * params.width + x;

    let cr = cur_re[idx];
    let ci = cur_im[idx];

    if params.is_first_frame == 1u {
        // Initialize: store current as previous, pass through unmodified
        prev_re[idx] = cr;
        prev_im[idx] = ci;
        lp_high[idx] = 0.0;
        lp_low[idx] = 0.0;
        out_re[idx] = cr;
        out_im[idx] = ci;
        return;
    }

    let pr = prev_re[idx];
    let pi_ = prev_im[idx]; // pi is not reserved but pi_ avoids confusion

    // Phase difference via conjugate product: current * conj(previous)
    // dot  = Re(current * conj(prev)) = cr*pr + ci*pi  = |z|² cos(Δθ)
    // cross = Im(current * conj(prev)) = ci*pr - cr*pi  = |z|² sin(Δθ)
    let dot = cr * pr + ci * pi_;
    let cross = ci * pr - cr * pi_;
    let phase_diff = atan2(cross, dot);

    // IIR temporal bandpass
    let new_lp_high = lp_high[idx] + params.alpha_high * (phase_diff - lp_high[idx]);
    let new_lp_low = lp_low[idx] + params.alpha_low * (phase_diff - lp_low[idx]);
    lp_high[idx] = new_lp_high;
    lp_low[idx] = new_lp_low;
    let filtered = new_lp_high - new_lp_low;

    // Amplify the filtered phase
    let amp_phase = params.amplification * filtered;

    // Rotate current sub-band by amplified phase
    let cos_p = cos(amp_phase);
    let sin_p = sin(amp_phase);
    out_re[idx] = cr * cos_p - ci * sin_p;
    out_im[idx] = cr * sin_p + ci * cos_p;

    // Store current as previous for next frame
    prev_re[idx] = cr;
    prev_im[idx] = ci;
}

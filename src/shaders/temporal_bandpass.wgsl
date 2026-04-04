// temporal_bandpass.wgsl — IIR temporal bandpass filter + amplification
//
// Eulerian Video Magnification core: for each pixel/channel, maintain two
// IIR lowpass filters at different cutoff frequencies. The difference is
// the bandpass signal (motion in the target frequency range). Amplify it
// and add back to the original frame.
//
// IIR update: lowpass[t] = alpha * input[t] + (1 - alpha) * lowpass[t-1]
// Bandpass = lowpass_high - lowpass_low
// Output = original + amplification * bandpass

struct Params {
    width: u32,
    height: u32,
    alpha_low: f32,     // IIR coefficient for low-frequency cutoff
    alpha_high: f32,    // IIR coefficient for high-frequency cutoff
    amplification: f32, // magnification factor
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;         // current frame CHW
@group(0) @binding(2) var<storage, read_write> lp_high: array<f32>; // IIR state (high cutoff)
@group(0) @binding(3) var<storage, read_write> lp_low: array<f32>;  // IIR state (low cutoff)
@group(0) @binding(4) var<storage, read_write> output: array<f32>;  // output CHW

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height { return; }

    let hw = params.height * params.width;
    let idx = y * params.width + x;

    for (var c = 0u; c < 3u; c++) {
        let i = c * hw + idx;
        let val = input[i];

        // Update IIR lowpass filters
        let new_lp_high = params.alpha_high * val + (1.0 - params.alpha_high) * lp_high[i];
        let new_lp_low = params.alpha_low * val + (1.0 - params.alpha_low) * lp_low[i];

        lp_high[i] = new_lp_high;
        lp_low[i] = new_lp_low;

        // Bandpass = difference of lowpasses
        let bandpass = new_lp_high - new_lp_low;

        // Amplify and add back to original, clamp to [0, 1]
        output[i] = clamp(val + params.amplification * bandpass, 0.0, 1.0);
    }
}

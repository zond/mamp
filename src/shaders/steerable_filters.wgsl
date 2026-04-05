// steerable_filters.wgsl — Apply steerable pyramid filter in frequency domain
//
// Multiplies a complex 2D spectrum by a radial × angular filter to extract
// one oriented sub-band. The filter is computed analytically per pixel from
// the frequency coordinates, not stored as a lookup table.
//
// Simoncelli & Freeman steerable pyramid filter design:
//   Radial:  raised-cosine log-radial filter for each scale
//   Angular: cos^(K-1)(theta - theta_k) for K orientations
//
// Dispatch: (ceil(width/16), ceil(height/16), 1)

struct Params {
    width: u32,       // padded FFT width
    height: u32,      // padded FFT height
    num_orientations: u32,  // K (typically 4)
    orientation_idx: u32,   // which orientation (0..K-1)
    scale: u32,       // which scale (0 = finest)
    num_scales: u32,  // total number of scales
    filter_type: u32, // 0 = bandpass sub-band, 1 = highpass residual, 2 = lowpass residual
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_re: array<f32>;
@group(0) @binding(2) var<storage, read> input_im: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_re: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_im: array<f32>;

const PI: f32 = 3.14159265358979;

// Raised-cosine log-radial function used in steerable pyramid.
// Maps log-radius to a smooth [0,1] window.
fn raised_cosine(val: f32, center: f32, width: f32) -> f32 {
    let x = (val - center) / width;
    if x < -0.5 || x > 0.5 { return 0.0; }
    return 0.5 * (1.0 + cos(2.0 * PI * x));
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height { return; }

    let idx = y * params.width + x;
    let w = f32(params.width);
    let h = f32(params.height);

    // Frequency coordinates centered at (0,0) with DC at corner (standard FFT layout)
    // Map pixel (x, y) to frequency (-0.5..0.5)
    var fx = f32(x) / w;
    var fy = f32(y) / h;
    if fx > 0.5 { fx -= 1.0; }
    if fy > 0.5 { fy -= 1.0; }

    let radius = sqrt(fx * fx + fy * fy);
    let angle = atan2(fy, fx);

    var filter_val = 0.0;

    if params.filter_type == 1u {
        // Highpass residual: passes frequencies above the finest scale
        let log_rad = log2(max(radius, 1e-10));
        let hi_cutoff = -1.0; // log2(0.5) — Nyquist
        let bandwidth = 1.0;
        filter_val = raised_cosine(log_rad, hi_cutoff + 0.5, bandwidth);
        // Clamp to 1 above cutoff
        if log_rad > hi_cutoff + 0.5 { filter_val = 1.0; }

    } else if params.filter_type == 2u {
        // Lowpass residual: passes frequencies below the coarsest scale
        let log_rad = log2(max(radius, 1e-10));
        let scale_f = f32(params.num_scales);
        let lo_cutoff = -1.0 - scale_f;
        let bandwidth = 1.0;
        filter_val = raised_cosine(log_rad, lo_cutoff - 0.5, bandwidth);
        if log_rad < lo_cutoff - 0.5 { filter_val = 1.0; }

    } else {
        // Bandpass sub-band: radial × angular

        // Radial component: octave-bandwidth raised cosine
        let log_rad = log2(max(radius, 1e-10));
        let center_log_rad = -1.0 - f32(params.scale); // -1 for finest, -2 for next, etc.
        let radial = raised_cosine(log_rad, center_log_rad, 1.0);

        // Angular component: cos^(K-1)(angle - target_angle) with proper normalization
        let k = f32(params.num_orientations);
        let target_angle = PI * f32(params.orientation_idx) / k;
        let da = angle - target_angle;

        // Wrap angle difference to [-pi, pi]
        var wrapped = da;
        if wrapped > PI { wrapped -= 2.0 * PI; }
        if wrapped < -PI { wrapped += 2.0 * PI; }

        // Angular selectivity: cos^(K-1)(angle) with normalization
        let cos_da = cos(wrapped);
        // Only positive lobe (half-plane)
        var angular = 0.0;
        if abs(wrapped) < PI / 2.0 {
            angular = pow(cos_da, k - 1.0);
        }

        // Normalization factor for angular component
        // For K orientations: sum over all orientations = 1 at any angle
        // Norm = (2^(K-1) * (K-1)!) / K!  (Simoncelli)
        // For K=4: norm = 8*6/24 = 2.0  → multiply by 2
        // General: 2^(K-1) / binomial(K-1, (K-1)/2) ... simplified:
        let norm = sqrt(k) * 1.4142135; // approximate normalization
        angular *= norm;

        filter_val = radial * angular;
    }

    // Handle DC (radius=0): only lowpass gets it
    if x == 0u && y == 0u && params.filter_type != 2u {
        filter_val = 0.0;
    }

    // Complex multiply: output = input * filter (filter is real-valued)
    output_re[idx] = input_re[idx] * filter_val;
    output_im[idx] = input_im[idx] * filter_val;
}

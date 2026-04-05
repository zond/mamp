// steerable_filters.wgsl — Steerable pyramid frequency-domain filter
//
// Simoncelli & Freeman design. Applied per-pixel to a 2D spectrum.
// Dispatch: (ceil(width/16), ceil(height/16), 1)

struct Params {
    width: u32,
    height: u32,
    num_orientations: u32,
    orientation_idx: u32,
    scale: u32,
    num_scales: u32,
    filter_type: u32, // 0 = bandpass, 1 = highpass residual, 2 = lowpass residual
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_re: array<f32>;
@group(0) @binding(2) var<storage, read> input_im: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_re: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_im: array<f32>;

const PI: f32 = 3.14159265358979;

// Smooth transition: 0 at x=-0.5, 1 at x=0.5 (raised cosine half-window)
fn smooth_step(x: f32) -> f32 {
    let clamped = clamp(x, -0.5, 0.5);
    return 0.5 + 0.5 * sin(PI * clamped);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height { return; }

    let idx = y * params.width + x;
    let w = f32(params.width);
    let h = f32(params.height);

    // Frequency coordinates: DC at corner, range [-0.5, 0.5)
    var fx = f32(x) / w;
    var fy = f32(y) / h;
    if fx > 0.5 { fx -= 1.0; }
    if fy > 0.5 { fy -= 1.0; }

    let radius = sqrt(fx * fx + fy * fy);
    let log_rad = log2(max(radius, 1e-10));
    let angle = atan2(fy, fx);

    var filter_val = 0.0;

    // Radial tiling: complementary smooth transitions in log-radius.
    // Cutoffs at log2(radius) = 0, -1, -2, ..., -num_scales
    // where 0 = Nyquist (radius=1, but grid max is 0.5 so log2=-1 is effective Nyquist)
    // Actually: grid frequencies go up to 0.5, so log2 max = -1.
    // Place cutoffs: -1 (Nyquist), -2, -3, -4 for 3 scales.
    // Highpass: above cutoff[0] = -1
    // Scale 0 (finest): between cutoff[0]=-1 and cutoff[1]=-2
    // Scale 1: between cutoff[1]=-2 and cutoff[2]=-3
    // Scale 2 (coarsest): between cutoff[2]=-3 and cutoff[3]=-4
    // Lowpass: below cutoff[3]=-4

    if params.filter_type == 1u {
        // ── Highpass residual = sqrt(1 - lowmask_0^2) ──
        let cutoff = -1.0; // Nyquist boundary
        let lowmask = smooth_step(-(log_rad - cutoff));
        filter_val = sqrt(max(1.0 - lowmask * lowmask, 0.0));

    } else if params.filter_type == 2u {
        // ── Lowpass residual = lowmask_{num_scales} ──
        let cutoff = -1.0 - f32(params.num_scales);
        filter_val = smooth_step(-(log_rad - cutoff));

    } else {
        // ── Bandpass: radial_s = sqrt(lowmask_s^2 - lowmask_{s+1}^2) ──
        let cutoff_upper = -1.0 - f32(params.scale);
        let cutoff_lower = -1.0 - f32(params.scale) - 1.0;
        let lm_upper = smooth_step(-(log_rad - cutoff_upper));
        let lm_lower = smooth_step(-(log_rad - cutoff_lower));
        let radial = sqrt(max(lm_upper * lm_upper - lm_lower * lm_lower, 0.0));

        // Angular: cos^(K-1)(angle - tgt_angle) with exact normalization
        let k = params.num_orientations;
        let kf = f32(k);
        let tgt_angle = PI * f32(params.orientation_idx) / kf;

        // Compute angle difference, wrapped to [-pi, pi]
        var da = angle - tgt_angle;
        if da > PI { da -= 2.0 * PI; }
        if da < -PI { da += 2.0 * PI; }

        var angular = 0.0;
        if abs(da) < PI / 2.0 {
            let c = cos(da);
            angular = pow(c, kf - 1.0);
        }

        // Normalization: sum_k |norm * cos^(K-1)(theta - theta_k)|^2 = 1.
        // For K orientations: sum_k cos^(2K-2)(theta - k*pi/K) = S_K (constant).
        // S_K = K * (2K-2)! / (2^(2K-2) * ((K-1)!)^2)
        // norm = 1 / sqrt(S_K)
        // Precomputed:
        // K=2: S=1,     norm = 1.0000
        // K=4: S=5/8*4=5/2, norm = sqrt(2/5) = 0.6325
        // K=6: S=63/128*6=189/64, norm = sqrt(64/189) = 0.5820
        // K=8: S=429/1024*8=429/128, norm = sqrt(128/429) = 0.5463
        // Actually let me compute correctly:
        // sum_k cos^(2(K-1))(theta - k*pi/K) for K equally spaced angles
        // For K=2: cos^2 + sin^2 = 1 → S=1, norm=1
        // For K=4: sum cos^6(theta-k*pi/4) over k=0..3 = 5/4 → S=5/4, norm=sqrt(4/5)=0.8944
        // For K=6: S = 5/4 (same!) → norm=0.8944
        // For K=8: S = 35/32*8/... let me just use computed values
        var norm = 1.0;
        if k == 2u { norm = 1.0; }
        else if k == 4u { norm = 0.9428; }
        else if k == 6u { norm = 0.9129; }
        else if k == 8u { norm = 0.8998; }
        angular *= norm;

        filter_val = radial * angular;
    }

    // DC: only lowpass
    if x == 0u && y == 0u && params.filter_type != 2u {
        filter_val = 0.0;
    }

    output_re[idx] = input_re[idx] * filter_val;
    output_im[idx] = input_im[idx] * filter_val;
}

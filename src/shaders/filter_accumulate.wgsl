// filter_accumulate.wgsl — Apply steerable filter and accumulate into output spectrum
//
// Two modes:
//   mode=0: output += input × filter       (for phase-modified sub-bands)
//   mode=1: output += input × filter²      (for residuals that skip phase processing)
//
// Uses the same filter computation as steerable_filters.wgsl.

struct Params {
    width: u32,
    height: u32,
    num_orientations: u32,
    orientation_idx: u32,
    scale: u32,
    num_scales: u32,
    filter_type: u32, // 0=bandpass, 1=highpass, 2=lowpass
    mode: u32,        // 0=×filter, 1=×filter²
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_re: array<f32>;
@group(0) @binding(2) var<storage, read> input_im: array<f32>;
@group(0) @binding(3) var<storage, read_write> accum_re: array<f32>;
@group(0) @binding(4) var<storage, read_write> accum_im: array<f32>;

const PI: f32 = 3.14159265358979;

fn smooth_step(x: f32) -> f32 {
    let clamped = clamp(x, -0.5, 0.5);
    return 0.5 + 0.5 * sin(PI * clamped);
}

fn compute_filter(x: u32, y: u32) -> f32 {
    let w = f32(params.width);
    let h = f32(params.height);

    var fx = f32(x) / w;
    var fy = f32(y) / h;
    if fx > 0.5 { fx -= 1.0; }
    if fy > 0.5 { fy -= 1.0; }

    let radius = sqrt(fx * fx + fy * fy);
    let log_rad = log2(max(radius, 1e-10));
    let angle = atan2(fy, fx);

    var filter_val = 0.0;

    if params.filter_type == 1u {
        let cutoff = -1.0;
        let lowmask = smooth_step(-(log_rad - cutoff));
        filter_val = sqrt(max(1.0 - lowmask * lowmask, 0.0));
    } else if params.filter_type == 2u {
        let cutoff = -1.0 - f32(params.num_scales);
        filter_val = smooth_step(-(log_rad - cutoff));
    } else {
        let cutoff_upper = -1.0 - f32(params.scale);
        let cutoff_lower = -1.0 - f32(params.scale) - 1.0;
        let lm_upper = smooth_step(-(log_rad - cutoff_upper));
        let lm_lower = smooth_step(-(log_rad - cutoff_lower));
        let radial = sqrt(max(lm_upper * lm_upper - lm_lower * lm_lower, 0.0));

        let k = params.num_orientations;
        let kf = f32(k);
        let tgt_angle = PI * f32(params.orientation_idx) / kf;
        var da = angle - tgt_angle;
        if da > PI { da -= 2.0 * PI; }
        if da < -PI { da += 2.0 * PI; }

        var angular = 0.0;
        if abs(da) < PI / 2.0 {
            angular = pow(cos(da), kf - 1.0);
        }

        var norm = 1.0;
        if k == 2u { norm = 1.0; }
        else if k == 4u { norm = 0.9428; }
        else if k == 6u { norm = 0.9129; }
        else if k == 8u { norm = 0.8998; }
        angular *= norm;

        filter_val = radial * angular;
    }

    if x == 0u && y == 0u && params.filter_type != 2u {
        filter_val = 0.0;
    }

    return filter_val;
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= params.width || y >= params.height { return; }

    let idx = y * params.width + x;
    var h = compute_filter(x, y);

    if params.mode == 1u {
        h = h * h; // squared for residuals
    }

    accum_re[idx] += input_re[idx] * h;
    accum_im[idx] += input_im[idx] * h;
}

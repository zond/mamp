// upsample.wgsl — Bilinear upsampling for decoder (optimized)
// Ha et al. key finding: reducing spatial resolution of latent motion representation
// in the decoder provides good efficiency/quality trade-off.
//
// OPTIMIZATION: Each thread processes 4 consecutive channels at the same output (x, y).
// The coordinate mapping and clamping are computed once, then reused across all 4 channels.
// Channel data is loaded and interpolated using vec4 SIMD operations since the 4 input
// samples at (c, y0, x0), (c, y0, x1), (c, y1, x0), (c, y1, x1) share the same spatial
// offsets across channels in CHW layout.
//
// DISPATCH CHANGE: z-dimension is now ceil(channels / 4) instead of channels.
//   Old: dispatch(ceil(out_W/8), ceil(out_H/8), channels)
//   New: dispatch(ceil(out_W/8), ceil(out_H/8), ceil(channels/4))

struct UpsampleParams {
    channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> params: UpsampleParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let c_base = gid.z * 4u;

    if ox >= params.out_width || oy >= params.out_height || c_base >= params.channels {
        return;
    }

    // Map output coords to input coords (shared across all channels)
    let scale_x = f32(params.in_width) / f32(params.out_width);
    let scale_y = f32(params.in_height) / f32(params.out_height);

    let src_x = (f32(ox) + 0.5) * scale_x - 0.5;
    let src_y = (f32(oy) + 0.5) * scale_y - 0.5;

    let x0 = u32(max(floor(src_x), 0.0));
    let y0 = u32(max(floor(src_y), 0.0));
    let x1 = min(x0 + 1u, params.in_width - 1u);
    let y1 = min(y0 + 1u, params.in_height - 1u);

    let fx = src_x - floor(src_x);
    let fy = src_y - floor(src_y);

    // Precompute interpolation weights
    let w00 = (1.0 - fx) * (1.0 - fy);
    let w01 = fx * (1.0 - fy);
    let w10 = (1.0 - fx) * fy;
    let w11 = fx * fy;

    let in_hw = params.in_height * params.in_width;
    let out_hw = params.out_height * params.out_width;

    // Spatial offsets into a channel plane (same for all channels)
    let s00 = y0 * params.in_width + x0;
    let s01 = y0 * params.in_width + x1;
    let s10 = y1 * params.in_width + x0;
    let s11 = y1 * params.in_width + x1;

    let out_spatial = oy * params.out_width + ox;

    let c_remaining = params.channels - c_base;

    if c_remaining >= 4u {
        // Fast path: process 4 channels with vec4 SIMD
        let plane0 = c_base * in_hw;
        let plane1 = plane0 + in_hw;
        let plane2 = plane0 + 2u * in_hw;
        let plane3 = plane0 + 3u * in_hw;

        let v00 = vec4<f32>(input[plane0 + s00], input[plane1 + s00], input[plane2 + s00], input[plane3 + s00]);
        let v01 = vec4<f32>(input[plane0 + s01], input[plane1 + s01], input[plane2 + s01], input[plane3 + s01]);
        let v10 = vec4<f32>(input[plane0 + s10], input[plane1 + s10], input[plane2 + s10], input[plane3 + s10]);
        let v11 = vec4<f32>(input[plane0 + s11], input[plane1 + s11], input[plane2 + s11], input[plane3 + s11]);

        let val = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;

        let out0 = c_base * out_hw + out_spatial;
        output[out0]            = val.x;
        output[out0 + out_hw]   = val.y;
        output[out0 + 2u * out_hw] = val.z;
        output[out0 + 3u * out_hw] = val.w;
    } else {
        // Tail: process remaining 1-3 channels individually
        for (var dc = 0u; dc < c_remaining; dc = dc + 1u) {
            let c = c_base + dc;
            let plane = c * in_hw;

            let v00 = input[plane + s00];
            let v01 = input[plane + s01];
            let v10 = input[plane + s10];
            let v11 = input[plane + s11];

            let val = v00 * w00 + v01 * w01 + v10 * w10 + v11 * w11;

            output[c * out_hw + out_spatial] = val;
        }
    }
}

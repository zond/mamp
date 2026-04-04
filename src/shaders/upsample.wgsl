// upsample.wgsl — Bilinear upsampling for decoder
// Ha et al. key finding: reducing spatial resolution of latent motion representation
// in the decoder provides good efficiency/quality trade-off.

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

fn in_idx(c: u32, y: u32, x: u32) -> u32 {
    return c * params.in_height * params.in_width + y * params.in_width + x;
}

fn out_idx(c: u32, y: u32, x: u32) -> u32 {
    return c * params.out_height * params.out_width + y * params.out_width + x;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let c = gid.z;

    if ox >= params.out_width || oy >= params.out_height || c >= params.channels {
        return;
    }

    // Map output coords to input coords
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

    let v00 = input[in_idx(c, y0, x0)];
    let v10 = input[in_idx(c, y1, x0)];
    let v01 = input[in_idx(c, y0, x1)];
    let v11 = input[in_idx(c, y1, x1)];

    let val = v00 * (1.0 - fx) * (1.0 - fy)
            + v01 * fx * (1.0 - fy)
            + v10 * (1.0 - fx) * fy
            + v11 * fx * fy;

    output[out_idx(c, oy, ox)] = val;
}

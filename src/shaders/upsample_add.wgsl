// upsample_add.wgsl — Bilinear upsample coarser level + add finer level
// output[x,y] = fine[x,y] + bilinear_upsample(coarse, x, y)
// Used for collapsing the Laplacian pyramid during reconstruction.

struct Params {
    fine_width: u32,
    fine_height: u32,
    coarse_width: u32,
    coarse_height: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> fine: array<f32>;    // amplified Laplacian at this level
@group(0) @binding(2) var<storage, read> coarse: array<f32>;  // reconstruction from coarser level
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    if ox >= params.fine_width || oy >= params.fine_height { return; }

    let fine_hw = params.fine_height * params.fine_width;
    let coarse_hw = params.coarse_height * params.coarse_width;

    // Map fine pixel to coarse coordinates
    let scale_x = f32(params.coarse_width) / f32(params.fine_width);
    let scale_y = f32(params.coarse_height) / f32(params.fine_height);
    let src_x = (f32(ox) + 0.5) * scale_x - 0.5;
    let src_y = (f32(oy) + 0.5) * scale_y - 0.5;

    let x0 = u32(max(floor(src_x), 0.0));
    let y0 = u32(max(floor(src_y), 0.0));
    let x1 = min(x0 + 1u, params.coarse_width - 1u);
    let y1 = min(y0 + 1u, params.coarse_height - 1u);
    let fx = src_x - floor(src_x);
    let fy = src_y - floor(src_y);

    let fine_idx = oy * params.fine_width + ox;

    for (var c = 0u; c < 3u; c++) {
        let cb = c * coarse_hw;
        let v00 = coarse[cb + y0 * params.coarse_width + x0];
        let v10 = coarse[cb + y0 * params.coarse_width + x1];
        let v01 = coarse[cb + y1 * params.coarse_width + x0];
        let v11 = coarse[cb + y1 * params.coarse_width + x1];
        let upsampled = v00 * (1.0 - fx) * (1.0 - fy) + v10 * fx * (1.0 - fy) +
                         v01 * (1.0 - fx) * fy + v11 * fx * fy;

        output[c * fine_hw + fine_idx] = fine[c * fine_hw + fine_idx] + upsampled;
    }
}

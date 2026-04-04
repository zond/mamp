// gaussian_downsample.wgsl — 2×2 average downsample for Gaussian pyramid
// Input: CHW float at (in_width, in_height), Output: CHW float at (in_width/2, in_height/2)

struct Params {
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    if ox >= params.out_width || oy >= params.out_height { return; }

    let in_hw = params.in_height * params.in_width;
    let out_hw = params.out_height * params.out_width;
    let ix = ox * 2u;
    let iy = oy * 2u;
    let ix1 = min(ix + 1u, params.in_width - 1u);
    let iy1 = min(iy + 1u, params.in_height - 1u);

    for (var c = 0u; c < 3u; c++) {
        let base = c * in_hw;
        let v00 = input[base + iy * params.in_width + ix];
        let v10 = input[base + iy * params.in_width + ix1];
        let v01 = input[base + iy1 * params.in_width + ix];
        let v11 = input[base + iy1 * params.in_width + ix1];
        output[c * out_hw + oy * params.out_width + ox] = 0.25 * (v00 + v10 + v01 + v11);
    }
}

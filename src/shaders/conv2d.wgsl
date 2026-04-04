// conv2d.wgsl — General-purpose Conv2D + ReLU compute shader

struct ConvParams {
    in_channels: u32,
    out_channels: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    width: u32,
    height: u32,
    use_relu: u32,
}

@group(0) @binding(0) var<uniform> params: ConvParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

fn idx3(c: u32, y: u32, x: u32, h: u32, w: u32) -> u32 {
    return c * h * w + y * w + x;
}

fn idx4(oc: u32, ic: u32, ky: u32, kx: u32, ic_count: u32, k: u32) -> u32 {
    return oc * (ic_count * k * k) + ic * (k * k) + ky * k + kx;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ox = gid.x;
    let oy = gid.y;
    let oc = gid.z;

    let out_w = (params.width + 2u * params.padding - params.kernel_size) / params.stride + 1u;
    let out_h = (params.height + 2u * params.padding - params.kernel_size) / params.stride + 1u;

    if ox >= out_w || oy >= out_h || oc >= params.out_channels {
        return;
    }

    var sum: f32 = bias[oc];

    for (var ic: u32 = 0u; ic < params.in_channels; ic++) {
        for (var ky: u32 = 0u; ky < params.kernel_size; ky++) {
            for (var kx: u32 = 0u; kx < params.kernel_size; kx++) {
                let iy = i32(oy * params.stride + ky) - i32(params.padding);
                let ix = i32(ox * params.stride + kx) - i32(params.padding);

                if iy >= 0 && iy < i32(params.height) && ix >= 0 && ix < i32(params.width) {
                    let in_val = input[idx3(ic, u32(iy), u32(ix), params.height, params.width)];
                    let w_val = weights[idx4(oc, ic, ky, kx, params.in_channels, params.kernel_size)];
                    sum += in_val * w_val;
                }
            }
        }
    }

    if params.use_relu == 1u {
        sum = max(sum, 0.0);
    }

    output[idx3(oc, oy, ox, out_h, out_w)] = sum;
}

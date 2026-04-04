// manipulator.wgsl — Motion representation manipulation (optimized)
// Takes encoded representations of two frames, computes motion diff, amplifies by alpha.
// Ha et al. key insight: a single linear layer suffices for manipulation.
//
// OPTIMIZATION: Each thread processes 4 consecutive channels at the same (x, y) position.
// Since CHW layout stores channels as contiguous planes of H*W, four consecutive channel
// planes at the same spatial position are strided by H*W. We load all four at once,
// compute via vec4 SIMD, and store in one shot.
//
// DISPATCH CHANGE: z-dimension is now ceil(channels / 4) instead of channels.
//   Old: dispatch(ceil(W/8), ceil(H/8), channels)
//   New: dispatch(ceil(W/8), ceil(H/8), ceil(channels/4))

struct ManipParams {
    channels: u32,
    height: u32,
    width: u32,
    alpha: f32,  // magnification factor
}

@group(0) @binding(0) var<uniform> params: ManipParams;
@group(0) @binding(1) var<storage, read> shape_rep_a: array<f32>;   // shape representation frame A
@group(0) @binding(2) var<storage, read> shape_rep_b: array<f32>;   // shape representation frame B
@group(0) @binding(3) var<storage, read> texture_rep: array<f32>;   // texture representation (from frame A)
@group(0) @binding(4) var<storage, read_write> output: array<f32>;  // magnified motion representation

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let c_base = gid.z * 4u;  // each thread handles 4 channels starting here

    if x >= params.width || y >= params.height || c_base >= params.channels {
        return;
    }

    let hw = params.height * params.width;
    let spatial = y * params.width + x;

    // Determine how many channels this thread actually processes (1..4)
    let c_remaining = params.channels - c_base;

    // Base index for the first channel this thread handles
    let base = c_base * hw + spatial;

    if c_remaining >= 4u {
        // Fast path: full vec4 of 4 channels
        let i0 = base;
        let i1 = base + hw;
        let i2 = base + 2u * hw;
        let i3 = base + 3u * hw;

        let a = vec4<f32>(shape_rep_a[i0], shape_rep_a[i1], shape_rep_a[i2], shape_rep_a[i3]);
        let b = vec4<f32>(shape_rep_b[i0], shape_rep_b[i1], shape_rep_b[i2], shape_rep_b[i3]);
        let t = vec4<f32>(texture_rep[i0], texture_rep[i1], texture_rep[i2], texture_rep[i3]);

        let result = t + params.alpha * (b - a);

        output[i0] = result.x;
        output[i1] = result.y;
        output[i2] = result.z;
        output[i3] = result.w;
    } else if c_remaining == 3u {
        let i0 = base;
        let i1 = base + hw;
        let i2 = base + 2u * hw;

        let a = vec3<f32>(shape_rep_a[i0], shape_rep_a[i1], shape_rep_a[i2]);
        let b = vec3<f32>(shape_rep_b[i0], shape_rep_b[i1], shape_rep_b[i2]);
        let t = vec3<f32>(texture_rep[i0], texture_rep[i1], texture_rep[i2]);

        let result = t + params.alpha * (b - a);

        output[i0] = result.x;
        output[i1] = result.y;
        output[i2] = result.z;
    } else if c_remaining == 2u {
        let i0 = base;
        let i1 = base + hw;

        let a = vec2<f32>(shape_rep_a[i0], shape_rep_a[i1]);
        let b = vec2<f32>(shape_rep_b[i0], shape_rep_b[i1]);
        let t = vec2<f32>(texture_rep[i0], texture_rep[i1]);

        let result = t + params.alpha * (b - a);

        output[i0] = result.x;
        output[i1] = result.y;
    } else {
        // c_remaining == 1
        let motion = shape_rep_b[base] - shape_rep_a[base];
        output[base] = texture_rep[base] + params.alpha * motion;
    }
}

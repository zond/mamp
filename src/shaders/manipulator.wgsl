// manipulator.wgsl — Motion representation manipulation
// Takes encoded representations of two frames, computes motion diff, amplifies by alpha.
// Ha et al. key insight: a single linear layer suffices for manipulation.

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

fn idx(c: u32, y: u32, x: u32) -> u32 {
    return c * params.height * params.width + y * params.width + x;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let c = gid.z;

    if x >= params.width || y >= params.height || c >= params.channels {
        return;
    }

    let i = idx(c, y, x);

    // Motion = difference in shape representations between frames
    let motion = shape_rep_b[i] - shape_rep_a[i];

    // Amplify motion by alpha, add back to texture
    // Y = texture + alpha * (shape_b - shape_a)
    output[i] = texture_rep[i] + params.alpha * motion;
}

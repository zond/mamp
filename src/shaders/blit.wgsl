// blit.wgsl — Fullscreen quad that reads packed RGBA u32 from a storage buffer
// and outputs to the render target. No vertex buffer needed — positions are
// generated from vertex_index.

struct Params {
    width: u32,
    height: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> pixels: array<u32>;

struct VertexOutput {
    @builtin(position) pos: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> VertexOutput {
    // Fullscreen triangle (3 vertices cover the entire screen)
    var out: VertexOutput;
    let x = f32(i32(vi & 1u) * 2 - 1);
    let y = f32(i32(vi >> 1u) * 2 - 1);
    // Two-triangle fullscreen quad: vertices 0,1,2 and 2,1,3
    let positions = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0),
        vec2(-1.0,  1.0), vec2(1.0, -1.0), vec2(1.0,  1.0),
    );
    let uvs = array<vec2<f32>, 6>(
        vec2(0.0, 1.0), vec2(1.0, 1.0), vec2(0.0, 0.0),
        vec2(0.0, 0.0), vec2(1.0, 1.0), vec2(1.0, 0.0),
    );
    out.pos = vec4(positions[vi], 0.0, 1.0);
    out.uv = uvs[vi];
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let x = min(u32(in.uv.x * f32(params.width)), params.width - 1u);
    let y = min(u32(in.uv.y * f32(params.height)), params.height - 1u);
    let idx = y * params.width + x;

    // Safety: check bounds (buffer might be smaller than expected)
    let buf_size = params.width * params.height;
    if idx >= buf_size {
        return vec4(1.0, 0.0, 0.0, 1.0); // red = out of bounds
    }

    let packed = pixels[idx];
    if packed == 0u {
        // DEBUG: green tint if pixel is zero (helps distinguish "no data" from "black pixel")
        return vec4(0.0, 0.05, 0.0, 1.0);
    }

    let r = f32(packed & 0xFFu) / 255.0;
    let g = f32((packed >> 8u) & 0xFFu) / 255.0;
    let b = f32((packed >> 16u) & 0xFFu) / 255.0;

    return vec4(r, g, b, 1.0);
}

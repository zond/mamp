// fft.wgsl — 1D FFT using workgroup shared memory
//
// Each workgroup processes one row of complex data.
// Complex values stored as interleaved (re, im, re, im, ...) in f32 buffers.
// Supports lengths up to 256 (one complex element per thread).
//
// Algorithm: iterative Cooley-Tukey with bit-reversal permutation in shared memory.
//
// Dispatch: (num_rows, 1, 1) with workgroup_size(256, 1, 1)
// For lengths < 256, excess threads exit early.

struct Params {
    n: u32,          // FFT length (must be power of 2, max 256)
    log2_n: u32,     // log2(n)
    num_rows: u32,   // number of rows to process
    inverse: u32,    // 0 = forward FFT, 1 = inverse FFT
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_re: array<f32>;
@group(0) @binding(2) var<storage, read> input_im: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_re: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_im: array<f32>;

// Shared memory for in-place FFT (max 256 complex values)
var<workgroup> shared_re: array<f32, 256>;
var<workgroup> shared_im: array<f32, 256>;

// Bit-reverse an index within log2_n bits
fn bit_reverse(x: u32, bits: u32) -> u32 {
    var v = x;
    var r = 0u;
    for (var i = 0u; i < bits; i++) {
        r = (r << 1u) | (v & 1u);
        v = v >> 1u;
    }
    return r;
}

@compute @workgroup_size(256, 1, 1)
fn main(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let tid = lid.x;            // thread index within workgroup (0..255)
    let row = wid.x;            // which row we're processing
    let n = params.n;
    let log2_n = params.log2_n;

    // Skip if thread index >= n or row >= num_rows
    if tid >= n || row >= params.num_rows {
        return;
    }

    let row_offset = row * n;

    // Load input into shared memory with bit-reversal permutation
    let rev_tid = bit_reverse(tid, log2_n);
    shared_re[rev_tid] = input_re[row_offset + tid];
    shared_im[rev_tid] = input_im[row_offset + tid];

    workgroupBarrier();

    // Butterfly stages
    let sign = select(-1.0, 1.0, params.inverse == 1u); // -1 for forward, +1 for inverse

    for (var stage = 0u; stage < log2_n; stage++) {
        let half_size = 1u << stage;        // number of butterflies in each group
        let full_size = half_size << 1u;     // size of each butterfly group

        let group = tid / half_size;
        let k = tid % half_size;
        let idx1 = group * full_size + k;
        let idx2 = idx1 + half_size;

        // Twiddle factor: W_N^k = exp(sign * 2*pi*i * k / full_size)
        let angle = sign * 2.0 * 3.14159265358979 * f32(k) / f32(full_size);
        let tw_re = cos(angle);
        let tw_im = sin(angle);

        // Read values
        let a_re = shared_re[idx1];
        let a_im = shared_im[idx1];
        let b_re = shared_re[idx2];
        let b_im = shared_im[idx2];

        // Complex multiply: t = b * twiddle
        let t_re = b_re * tw_re - b_im * tw_im;
        let t_im = b_re * tw_im + b_im * tw_re;

        workgroupBarrier();

        // Butterfly
        shared_re[idx1] = a_re + t_re;
        shared_im[idx1] = a_im + t_im;
        shared_re[idx2] = a_re - t_re;
        shared_im[idx2] = a_im - t_im;

        workgroupBarrier();
    }

    // Write output (with normalization for inverse FFT)
    let scale = select(1.0, 1.0 / f32(n), params.inverse == 1u);
    output_re[row_offset + tid] = shared_re[tid] * scale;
    output_im[row_offset + tid] = shared_im[tid] * scale;
}

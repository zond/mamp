// fft.wgsl — 1D FFT using workgroup shared memory
//
// Supports lengths up to MAX_N (set by replacing SHARED_SIZE token at compile time).
// SHARED_SIZE=2048 needs 16KB shared memory; SHARED_SIZE=1024 needs 8KB.
// 256 threads per workgroup, each handling up to SHARED_SIZE/256 elements.
//
// The stride/fft_stride params enable both row-wise and column-wise FFT:
//   Row FFT:    stride=1,     fft_stride=width
//   Column FFT: stride=width, fft_stride=1
//
// Dispatch: (num_ffts, 1, 1)

struct Params {
    n: u32,          // FFT length (power of 2, <= SHARED_SIZE)
    log2_n: u32,     // log2(n)
    num_ffts: u32,   // number of independent FFTs
    inverse: u32,    // 0 = forward, 1 = inverse
    stride: u32,     // element stride (1 for rows, width for columns)
    fft_stride: u32, // offset between consecutive FFTs (width for rows, 1 for cols)
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input_re: array<f32>;
@group(0) @binding(2) var<storage, read> input_im: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_re: array<f32>;
@group(0) @binding(4) var<storage, read_write> output_im: array<f32>;

// SHARED_SIZE is replaced at compile time: 1024 or 2048
var<workgroup> shared_re: array<f32, /*SHARED_SIZE*/>;
var<workgroup> shared_im: array<f32, /*SHARED_SIZE*/>;

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
    let tid = lid.x;
    let fft_idx = wid.x;
    let n = params.n;
    let log2_n = params.log2_n;

    if fft_idx >= params.num_ffts { return; }

    let base_offset = fft_idx * params.fft_stride;
    let elems_per_thread = max(1u, n / 256u);

    // Load with bit-reversal permutation
    for (var e = 0u; e < elems_per_thread; e++) {
        let src_idx = tid + e * 256u;
        if src_idx < n {
            let rev = bit_reverse(src_idx, log2_n);
            let addr = base_offset + src_idx * params.stride;
            shared_re[rev] = input_re[addr];
            shared_im[rev] = input_im[addr];
        }
    }

    workgroupBarrier();

    // Butterfly stages
    let sign = select(-1.0, 1.0, params.inverse == 1u);
    let half_n = n / 2u;
    let butterflies_per_thread = max(1u, half_n / 256u);

    for (var stage = 0u; stage < log2_n; stage++) {
        let half_size = 1u << stage;
        let full_size = half_size << 1u;

        for (var b = 0u; b < butterflies_per_thread; b++) {
            let butterfly_id = tid + b * 256u;
            if butterfly_id < half_n {
                let group = butterfly_id / half_size;
                let k = butterfly_id % half_size;
                let idx1 = group * full_size + k;
                let idx2 = idx1 + half_size;

                let angle = sign * 2.0 * 3.14159265358979 * f32(k) / f32(full_size);
                let tw_re = cos(angle);
                let tw_im = sin(angle);

                let a_re = shared_re[idx1];
                let a_im = shared_im[idx1];
                let b_re = shared_re[idx2];
                let b_im = shared_im[idx2];

                let t_re = b_re * tw_re - b_im * tw_im;
                let t_im = b_re * tw_im + b_im * tw_re;

                shared_re[idx1] = a_re + t_re;
                shared_im[idx1] = a_im + t_im;
                shared_re[idx2] = a_re - t_re;
                shared_im[idx2] = a_im - t_im;
            }
        }

        workgroupBarrier();
    }

    // Write output
    let scale = select(1.0, 1.0 / f32(n), params.inverse == 1u);
    for (var e = 0u; e < elems_per_thread; e++) {
        let dst_idx = tid + e * 256u;
        if dst_idx < n {
            let addr = base_offset + dst_idx * params.stride;
            output_re[addr] = shared_re[dst_idx] * scale;
            output_im[addr] = shared_im[dst_idx] * scale;
        }
    }
}

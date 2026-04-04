// evm.rs — Laplacian Pyramid Eulerian Video Magnification
//
// Pipeline per frame:
// 1. RGBA → CHW float
// 2. Build 4-level Gaussian pyramid (3 downsample dispatches)
// 3. Laplacian + temporal IIR bandpass at each level (4 dispatches)
// 4. Reconstruct pyramid (3 upsample+add dispatches)
// 5. CHW → RGBA + copy to staging
//
// Total: 12 dispatches in one encoder submission. No training, no weights.

use crate::gpu::GpuContext;
use bytemuck::{Pod, Zeroable};
use wgpu::*;

const N_LEVELS: usize = 4;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct FrameParams {
    width: u32,
    height: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct DownsampleParams {
    in_width: u32,
    in_height: u32,
    out_width: u32,
    out_height: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct LevelParams {
    width: u32,
    height: u32,
    coarse_width: u32,
    coarse_height: u32,
    alpha_low: f32,
    alpha_high: f32,
    amplification: f32,
    is_coarsest: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct UpsampleAddParams {
    fine_width: u32,
    fine_height: u32,
    coarse_width: u32,
    coarse_height: u32,
}

pub struct EvmPipeline {
    pub width: u32,
    pub height: u32,

    // Per-level dimensions
    level_w: [u32; N_LEVELS],
    level_h: [u32; N_LEVELS],

    // Pipelines
    rgba_to_chw_pipeline: ComputePipeline,
    chw_to_rgba_pipeline: ComputePipeline,
    downsample_pipeline: ComputePipeline,
    laplacian_temporal_pipeline: ComputePipeline,
    upsample_add_pipeline: ComputePipeline,

    // Frame I/O buffers
    buf_frame_rgba: Buffer,
    buf_output_rgba: Buffer,
    pub buf_staging: Buffer,

    // Gaussian pyramid: gaussian[0] = input CHW
    gaussian: [Buffer; N_LEVELS],

    // IIR temporal filter state (persistent across frames, levels 0..N_LEVELS-1)
    lp_high: [Buffer; N_LEVELS],
    lp_low: [Buffer; N_LEVELS],

    // Amplified Laplacian per level
    amplified: [Buffer; N_LEVELS],

    // Reconstruction output per level
    recon: [Buffer; N_LEVELS],

    // Pre-built uniform buffers
    frame_param_buf: Buffer,
    downsample_param_bufs: [Buffer; N_LEVELS - 1],
    level_param_bufs: [Buffer; N_LEVELS],
    upsample_add_param_bufs: [Buffer; N_LEVELS - 1],

    // Pre-built bind groups
    rgba_to_chw_bg: BindGroup,
    chw_to_rgba_bg: BindGroup,
    downsample_bgs: [BindGroup; N_LEVELS - 1],
    laplacian_temporal_bgs: [BindGroup; N_LEVELS],
    upsample_add_bgs: [BindGroup; N_LEVELS - 1],
}

impl EvmPipeline {
    pub fn new(ctx: &GpuContext, w: u32, h: u32) -> Self {
        assert!(w > 0 && h > 0);

        // Compute per-level dimensions
        let mut level_w = [0u32; N_LEVELS];
        let mut level_h = [0u32; N_LEVELS];
        level_w[0] = w;
        level_h[0] = h;
        for i in 1..N_LEVELS {
            level_w[i] = level_w[i - 1] / 2;
            level_h[i] = level_h[i - 1] / 2;
        }

        let s = BufferUsages::STORAGE;
        let scd = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        let pixels = (w * h) as u64;

        // Frame I/O
        let buf_frame_rgba = ctx.create_buffer("evm_frame", pixels * 4, scd);
        let buf_output_rgba = ctx.create_buffer("evm_out_rgba", pixels * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        let buf_staging = ctx.create_buffer("evm_staging", pixels * 4,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST);

        // Gaussian pyramid buffers
        let gaussian: [Buffer; N_LEVELS] = std::array::from_fn(|i| {
            let sz = 3 * (level_w[i] as u64) * (level_h[i] as u64) * 4;
            ctx.create_buffer(&format!("g{}", i), sz, s)
        });

        // IIR state + amplified + recon buffers per level
        let lp_high: [Buffer; N_LEVELS] = std::array::from_fn(|i| {
            let sz = 3 * (level_w[i] as u64) * (level_h[i] as u64) * 4;
            ctx.create_buffer(&format!("lph{}", i), sz, s)
        });
        let lp_low: [Buffer; N_LEVELS] = std::array::from_fn(|i| {
            let sz = 3 * (level_w[i] as u64) * (level_h[i] as u64) * 4;
            ctx.create_buffer(&format!("lpl{}", i), sz, s)
        });
        let amplified: [Buffer; N_LEVELS] = std::array::from_fn(|i| {
            let sz = 3 * (level_w[i] as u64) * (level_h[i] as u64) * 4;
            ctx.create_buffer(&format!("amp{}", i), sz, s)
        });
        let recon: [Buffer; N_LEVELS] = std::array::from_fn(|i| {
            let sz = 3 * (level_w[i] as u64) * (level_h[i] as u64) * 4;
            ctx.create_buffer(&format!("rec{}", i), sz, s)
        });

        // Pipelines
        let rgba_to_chw_pipeline = ctx.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("evm_r2c"), layout: None,
            module: &ctx.rgba_to_chw_module, entry_point: Some("main"),
            compilation_options: Default::default(), cache: None,
        });
        let chw_to_rgba_pipeline = ctx.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("evm_c2r"), layout: None,
            module: &ctx.chw_to_rgba_module, entry_point: Some("main"),
            compilation_options: Default::default(), cache: None,
        });

        let ds_module = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("downsample"),
            source: ShaderSource::Wgsl(include_str!("shaders/gaussian_downsample.wgsl").into()),
        });
        let downsample_pipeline = ctx.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("evm_ds"), layout: None,
            module: &ds_module, entry_point: Some("main"),
            compilation_options: Default::default(), cache: None,
        });

        let lt_module = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("laplacian_temporal"),
            source: ShaderSource::Wgsl(include_str!("shaders/laplacian_temporal.wgsl").into()),
        });
        let laplacian_temporal_pipeline = ctx.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("evm_lt"), layout: None,
            module: &lt_module, entry_point: Some("main"),
            compilation_options: Default::default(), cache: None,
        });

        let ua_module = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("upsample_add"),
            source: ShaderSource::Wgsl(include_str!("shaders/upsample_add.wgsl").into()),
        });
        let upsample_add_pipeline = ctx.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("evm_ua"), layout: None,
            module: &ua_module, entry_point: Some("main"),
            compilation_options: Default::default(), cache: None,
        });

        // Uniform buffers
        let frame_params = FrameParams { width: w, height: h, _pad0: 0, _pad1: 0 };
        let frame_param_buf = ctx.create_uniform("evm_fp", &frame_params);

        let downsample_param_bufs: [Buffer; N_LEVELS - 1] = std::array::from_fn(|i| {
            let p = DownsampleParams {
                in_width: level_w[i], in_height: level_h[i],
                out_width: level_w[i + 1], out_height: level_h[i + 1],
            };
            ctx.create_uniform(&format!("ds_p{}", i), &p)
        });

        // Level params will be updated each frame (alpha/freq change)
        let level_param_bufs: [Buffer; N_LEVELS] = std::array::from_fn(|i| {
            let p = LevelParams {
                width: level_w[i], height: level_h[i],
                coarse_width: if i + 1 < N_LEVELS { level_w[i + 1] } else { 0 },
                coarse_height: if i + 1 < N_LEVELS { level_h[i + 1] } else { 0 },
                alpha_low: 0.0, alpha_high: 0.0, amplification: 0.0,
                is_coarsest: if i == N_LEVELS - 1 { 1 } else { 0 },
            };
            ctx.create_uniform(&format!("lp{}", i), &p)
        });

        let upsample_add_param_bufs: [Buffer; N_LEVELS - 1] = std::array::from_fn(|i| {
            let p = UpsampleAddParams {
                fine_width: level_w[i], fine_height: level_h[i],
                coarse_width: level_w[i + 1], coarse_height: level_h[i + 1],
            };
            ctx.create_uniform(&format!("ua_p{}", i), &p)
        });

        // Bind groups
        let rgba_to_chw_bg = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("r2c_bg"),
            layout: &rgba_to_chw_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: frame_param_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: buf_frame_rgba.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: gaussian[0].as_entire_binding() },
            ],
        });

        let chw_to_rgba_bg = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("c2r_bg"),
            layout: &chw_to_rgba_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: frame_param_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: recon[0].as_entire_binding() },
                BindGroupEntry { binding: 2, resource: buf_output_rgba.as_entire_binding() },
            ],
        });

        let downsample_bgs: [BindGroup; N_LEVELS - 1] = std::array::from_fn(|i| {
            ctx.device.create_bind_group(&BindGroupDescriptor {
                label: Some(&format!("ds_bg{}", i)),
                layout: &downsample_pipeline.get_bind_group_layout(0),
                entries: &[
                    BindGroupEntry { binding: 0, resource: downsample_param_bufs[i].as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: gaussian[i].as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: gaussian[i + 1].as_entire_binding() },
                ],
            })
        });

        let laplacian_temporal_bgs: [BindGroup; N_LEVELS] = std::array::from_fn(|i| {
            let coarse_buf = if i + 1 < N_LEVELS { &gaussian[i + 1] } else { &gaussian[i] };
            ctx.device.create_bind_group(&BindGroupDescriptor {
                label: Some(&format!("lt_bg{}", i)),
                layout: &laplacian_temporal_pipeline.get_bind_group_layout(0),
                entries: &[
                    BindGroupEntry { binding: 0, resource: level_param_bufs[i].as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: gaussian[i].as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: coarse_buf.as_entire_binding() },
                    BindGroupEntry { binding: 3, resource: lp_high[i].as_entire_binding() },
                    BindGroupEntry { binding: 4, resource: lp_low[i].as_entire_binding() },
                    BindGroupEntry { binding: 5, resource: amplified[i].as_entire_binding() },
                ],
            })
        });

        let upsample_add_bgs: [BindGroup; N_LEVELS - 1] = std::array::from_fn(|i| {
            ctx.device.create_bind_group(&BindGroupDescriptor {
                label: Some(&format!("ua_bg{}", i)),
                layout: &upsample_add_pipeline.get_bind_group_layout(0),
                entries: &[
                    BindGroupEntry { binding: 0, resource: upsample_add_param_bufs[i].as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: amplified[i].as_entire_binding() },
                    BindGroupEntry { binding: 2, resource: recon[i + 1].as_entire_binding() },
                    BindGroupEntry { binding: 3, resource: recon[i].as_entire_binding() },
                ],
            })
        });

        Self {
            width: w, height: h, level_w, level_h,
            rgba_to_chw_pipeline, chw_to_rgba_pipeline,
            downsample_pipeline, laplacian_temporal_pipeline, upsample_add_pipeline,
            buf_frame_rgba, buf_output_rgba, buf_staging,
            gaussian, lp_high, lp_low, amplified, recon,
            frame_param_buf, downsample_param_bufs, level_param_bufs, upsample_add_param_bufs,
            rgba_to_chw_bg, chw_to_rgba_bg, downsample_bgs, laplacian_temporal_bgs, upsample_add_bgs,
        }
    }

    pub fn process_frame(
        &self,
        ctx: &GpuContext,
        frame_rgba: &[u32],
        amplification: f32,
        freq_low: f32,
        freq_high: f32,
        fps: f32,
    ) {
        let two_pi = 2.0 * std::f32::consts::PI;
        let alpha_low = 1.0 - (-two_pi * freq_low / fps).exp();
        let alpha_high = 1.0 - (-two_pi * freq_high / fps).exp();

        // Upload frame
        ctx.queue.write_buffer(&self.buf_frame_rgba, 0, bytemuck::cast_slice(frame_rgba));

        // Update per-level temporal params
        for i in 0..N_LEVELS {
            let p = LevelParams {
                width: self.level_w[i], height: self.level_h[i],
                coarse_width: if i + 1 < N_LEVELS { self.level_w[i + 1] } else { 0 },
                coarse_height: if i + 1 < N_LEVELS { self.level_h[i + 1] } else { 0 },
                alpha_low, alpha_high, amplification,
                is_coarsest: if i == N_LEVELS - 1 { 1 } else { 0 },
            };
            ctx.queue.write_buffer(&self.level_param_bufs[i], 0, bytemuck::bytes_of(&p));
        }

        let mut encoder = ctx.device.create_command_encoder(
            &CommandEncoderDescriptor { label: Some("evm") },
        );

        // 1. RGBA → CHW (into gaussian[0])
        Self::dispatch(&mut encoder, "r2c",
            &self.rgba_to_chw_pipeline, &self.rgba_to_chw_bg,
            GpuContext::div_ceil(self.width, 64),
            GpuContext::div_ceil(self.height, 16), 1);

        // 2. Gaussian pyramid: downsample levels 0→1, 1→2, 2→3
        for i in 0..(N_LEVELS - 1) {
            Self::dispatch(&mut encoder, "ds",
                &self.downsample_pipeline, &self.downsample_bgs[i],
                GpuContext::div_ceil(self.level_w[i + 1], 16),
                GpuContext::div_ceil(self.level_h[i + 1], 16), 1);
        }

        // 3. Laplacian + temporal bandpass at each level
        for i in 0..N_LEVELS {
            Self::dispatch(&mut encoder, "lt",
                &self.laplacian_temporal_pipeline, &self.laplacian_temporal_bgs[i],
                GpuContext::div_ceil(self.level_w[i], 16),
                GpuContext::div_ceil(self.level_h[i], 16), 1);
        }

        // 4. Reconstruct pyramid (coarsest to finest)
        // recon[N-1] = amplified[N-1] (copy coarsest residual)
        encoder.copy_buffer_to_buffer(
            &self.amplified[N_LEVELS - 1], 0,
            &self.recon[N_LEVELS - 1], 0,
            3 * (self.level_w[N_LEVELS - 1] as u64) * (self.level_h[N_LEVELS - 1] as u64) * 4,
        );
        // recon[i] = amplified[i] + upsample(recon[i+1])
        for i in (0..(N_LEVELS - 1)).rev() {
            Self::dispatch(&mut encoder, "ua",
                &self.upsample_add_pipeline, &self.upsample_add_bgs[i],
                GpuContext::div_ceil(self.level_w[i], 16),
                GpuContext::div_ceil(self.level_h[i], 16), 1);
        }

        // 5. CHW → RGBA
        Self::dispatch(&mut encoder, "c2r",
            &self.chw_to_rgba_pipeline, &self.chw_to_rgba_bg,
            GpuContext::div_ceil(self.width, 64),
            GpuContext::div_ceil(self.height, 16), 1);

        // 6. Copy to staging
        encoder.copy_buffer_to_buffer(
            &self.buf_output_rgba, 0,
            &self.buf_staging, 0,
            (self.width * self.height * 4) as u64,
        );

        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    fn dispatch(
        encoder: &mut CommandEncoder,
        label: &str,
        pipeline: &ComputePipeline,
        bg: &BindGroup,
        x: u32, y: u32, z: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        GpuContext::record_dispatch(&mut pass, pipeline, bg, (x, y, z));
    }
}

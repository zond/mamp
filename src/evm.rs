// evm.rs — Eulerian Video Magnification pipeline (no neural network)
//
// Simple temporal bandpass approach:
// 1. Convert RGBA → CHW float
// 2. IIR temporal bandpass filter per pixel (isolates motion in target frequency range)
// 3. Amplify bandpass signal and add back to original frame
// 4. Convert CHW → RGBA
//
// No training, no weights — just signal processing.

use crate::gpu::GpuContext;
use bytemuck::{Pod, Zeroable};
use wgpu::*;

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
struct BandpassParams {
    width: u32,
    height: u32,
    alpha_low: f32,
    alpha_high: f32,
    amplification: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

pub struct EvmPipeline {
    pub width: u32,
    pub height: u32,

    rgba_to_chw_pipeline: ComputePipeline,
    chw_to_rgba_pipeline: ComputePipeline,
    bandpass_pipeline: ComputePipeline,

    buf_frame_rgba: Buffer,
    buf_chw: Buffer,
    buf_lp_high: Buffer,
    buf_lp_low: Buffer,
    buf_output_chw: Buffer,
    buf_output_rgba: Buffer,
    pub buf_staging: Buffer,

    frame_param_buf: Buffer,
    bandpass_param_buf: Buffer,

    rgba_to_chw_bg: BindGroup,
    chw_to_rgba_bg: BindGroup,
    bandpass_bg: BindGroup,
}

impl EvmPipeline {
    pub fn new(ctx: &GpuContext, w: u32, h: u32) -> Self {
        assert!(w > 0 && h > 0);

        let pixels = (w * h) as u64;
        let chw_size = 3 * pixels * 4;
        let s = BufferUsages::STORAGE;
        let scd = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;

        let buf_frame_rgba = ctx.create_buffer("evm_frame", pixels * 4, scd);
        let buf_chw = ctx.create_buffer("evm_chw", chw_size, s);
        let buf_lp_high = ctx.create_buffer("evm_lp_high", chw_size, s);
        let buf_lp_low = ctx.create_buffer("evm_lp_low", chw_size, s);
        let buf_output_chw = ctx.create_buffer("evm_out_chw", chw_size, s);
        let buf_output_rgba = ctx.create_buffer("evm_out_rgba", pixels * 4,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        let buf_staging = ctx.create_buffer("evm_staging", pixels * 4,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST);

        let frame_params = FrameParams { width: w, height: h, _pad0: 0, _pad1: 0 };
        let frame_param_buf = ctx.create_uniform("evm_frame_params", &frame_params);

        let bandpass_params = BandpassParams {
            width: w, height: h,
            alpha_low: 0.0, alpha_high: 0.0,
            amplification: 0.0,
            _pad0: 0, _pad1: 0, _pad2: 0,
        };
        let bandpass_param_buf = ctx.create_uniform("evm_bp_params", &bandpass_params);

        // Pipelines
        let rgba_to_chw_pipeline = ctx.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("evm_rgba_to_chw"),
            layout: None,
            module: &ctx.rgba_to_chw_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let chw_to_rgba_pipeline = ctx.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("evm_chw_to_rgba"),
            layout: None,
            module: &ctx.chw_to_rgba_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let bandpass_module = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("temporal_bandpass"),
            source: ShaderSource::Wgsl(include_str!("shaders/temporal_bandpass.wgsl").into()),
        });

        let bandpass_pipeline = ctx.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("evm_bandpass"),
            layout: None,
            module: &bandpass_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        // Bind groups
        let rgba_to_chw_bg = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("evm_r2c_bg"),
            layout: &rgba_to_chw_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: frame_param_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: buf_frame_rgba.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: buf_chw.as_entire_binding() },
            ],
        });

        let chw_to_rgba_bg = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("evm_c2r_bg"),
            layout: &chw_to_rgba_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: frame_param_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: buf_output_chw.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: buf_output_rgba.as_entire_binding() },
            ],
        });

        let bandpass_bg = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("evm_bp_bg"),
            layout: &bandpass_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: bandpass_param_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: buf_chw.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: buf_lp_high.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: buf_lp_low.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: buf_output_chw.as_entire_binding() },
            ],
        });

        Self {
            width: w, height: h,
            rgba_to_chw_pipeline, chw_to_rgba_pipeline, bandpass_pipeline,
            buf_frame_rgba, buf_chw, buf_lp_high, buf_lp_low,
            buf_output_chw, buf_output_rgba, buf_staging,
            frame_param_buf, bandpass_param_buf,
            rgba_to_chw_bg, chw_to_rgba_bg, bandpass_bg,
        }
    }

    /// Process one frame. Call every frame with current camera RGBA pixels.
    /// `amplification`: magnification factor (e.g., 20.0)
    /// `freq_low`, `freq_high`: bandpass range in Hz (e.g., 0.5 - 3.0 for breathing)
    /// `fps`: current frame rate for computing IIR coefficients
    pub fn process_frame(
        &self,
        ctx: &GpuContext,
        frame_rgba: &[u32],
        amplification: f32,
        freq_low: f32,
        freq_high: f32,
        fps: f32,
    ) {
        let w = self.width;
        let h = self.height;

        // Compute IIR coefficients from frequencies
        // alpha = 1 - exp(-2*pi*freq/fps)
        let two_pi = 2.0 * std::f32::consts::PI;
        let alpha_low = 1.0 - (-two_pi * freq_low / fps).exp();
        let alpha_high = 1.0 - (-two_pi * freq_high / fps).exp();

        // Upload frame and update bandpass params
        ctx.queue.write_buffer(&self.buf_frame_rgba, 0, bytemuck::cast_slice(frame_rgba));

        let bp = BandpassParams {
            width: w, height: h,
            alpha_low, alpha_high,
            amplification,
            _pad0: 0, _pad1: 0, _pad2: 0,
        };
        ctx.queue.write_buffer(&self.bandpass_param_buf, 0, bytemuck::bytes_of(&bp));

        // Single encoder for all dispatches
        let mut encoder = ctx.device.create_command_encoder(
            &CommandEncoderDescriptor { label: Some("evm") },
        );

        // 1. RGBA → CHW
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("evm_r2c"),
                timestamp_writes: None,
            });
            GpuContext::record_dispatch(&mut pass,
                &self.rgba_to_chw_pipeline, &self.rgba_to_chw_bg,
                (GpuContext::div_ceil(w, 64), GpuContext::div_ceil(h, 16), 1));
        }

        // 2. Temporal bandpass + amplify
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("evm_bp"),
                timestamp_writes: None,
            });
            GpuContext::record_dispatch(&mut pass,
                &self.bandpass_pipeline, &self.bandpass_bg,
                (GpuContext::div_ceil(w, 16), GpuContext::div_ceil(h, 16), 1));
        }

        // 3. CHW → RGBA
        {
            let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
                label: Some("evm_c2r"),
                timestamp_writes: None,
            });
            GpuContext::record_dispatch(&mut pass,
                &self.chw_to_rgba_pipeline, &self.chw_to_rgba_bg,
                (GpuContext::div_ceil(w, 64), GpuContext::div_ceil(h, 16), 1));
        }

        // 4. Copy to staging for readback
        encoder.copy_buffer_to_buffer(
            &self.buf_output_rgba, 0,
            &self.buf_staging, 0,
            (w * h * 4) as u64,
        );

        ctx.queue.submit(std::iter::once(encoder.finish()));
    }
}

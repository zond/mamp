// model.rs — Ha et al. 2024 efficient motion magnification model
//
// Architecture (simplified from Oh et al. 2018):
//   Encoder:  Single branch, single linear layer  (their key finding #2)
//   Manipulator: motion_diff = shape_b - shape_a; out = texture + alpha * motion_diff
//   Decoder:  Reduced spatial resolution latent     (their key finding #1)
//
// Weight loading: expects raw f32 tensors exported from PyTorch via:
//   for name, param in model.state_dict().items():
//       param.cpu().numpy().tofile(f"weights/{name}.bin")

use crate::gpu::GpuContext;
use bytemuck::{Pod, Zeroable};
use wgpu::*;

// ── Uniform param structs (must match WGSL) ──────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ConvParams {
    pub in_channels: u32,
    pub out_channels: u32,
    pub kernel_size: u32,
    pub stride: u32,
    pub padding: u32,
    pub width: u32,
    pub height: u32,
    pub use_relu: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct ManipParams {
    pub channels: u32,
    pub height: u32,
    pub width: u32,
    pub alpha: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct UpsampleParams {
    pub channels: u32,
    pub in_height: u32,
    pub in_width: u32,
    pub out_height: u32,
    pub out_width: u32,
    pub _pad: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
pub struct FrameParams {
    pub width: u32,
    pub height: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

// ── Conv2D layer on GPU ──────────────────────────────────────────────────────

pub struct ConvLayer {
    pub pipeline: ComputePipeline,
    pub weight_buf: Buffer,
    pub bias_buf: Buffer,
    pub param_buf: Buffer,
    pub out_width: u32,
    pub out_height: u32,
    pub out_channels: u32,
}

impl ConvLayer {
    pub fn new(
        ctx: &GpuContext,
        label: &str,
        params: ConvParams,
        weights: &[f32],
        bias: &[f32],
    ) -> Self {
        let out_w = (params.width + 2 * params.padding - params.kernel_size) / params.stride + 1;
        let out_h = (params.height + 2 * params.padding - params.kernel_size) / params.stride + 1;

        let weight_buf = ctx.create_buffer_init(
            &format!("{}_weights", label),
            weights,
            BufferUsages::STORAGE,
        );
        let bias_buf = ctx.create_buffer_init(
            &format!("{}_bias", label),
            bias,
            BufferUsages::STORAGE,
        );
        let param_buf = ctx.create_uniform(&format!("{}_params", label), &params);

        let bgl = ctx.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some(&format!("{}_bgl", label)),
            entries: &[
                bgl_entry(0, ShaderStages::COMPUTE, true, false),  // params uniform
                bgl_entry(1, ShaderStages::COMPUTE, false, true),  // input storage read
                bgl_entry(2, ShaderStages::COMPUTE, false, true),  // weights storage read
                bgl_entry(3, ShaderStages::COMPUTE, false, true),  // bias storage read
                bgl_entry(4, ShaderStages::COMPUTE, false, false), // output storage rw
            ],
        });

        let pipeline_layout = ctx.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some(&format!("{}_layout", label)),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });

        let pipeline = ctx.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &ctx.conv2d_module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            weight_buf,
            bias_buf,
            param_buf,
            out_width: out_w,
            out_height: out_h,
            out_channels: params.out_channels,
        }
    }

    /// Run this conv layer. Caller provides input and output buffers.
    pub fn run(&self, ctx: &GpuContext, input: &Buffer, output: &Buffer) {
        let bind_group = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("conv_bg"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: self.param_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: self.weight_buf.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: self.bias_buf.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: output.as_entire_binding() },
            ],
        });

        let wg_x = GpuContext::div_ceil(self.out_width, 8);
        let wg_y = GpuContext::div_ceil(self.out_height, 8);
        let wg_z = self.out_channels;

        ctx.dispatch(&self.pipeline, &bind_group, (wg_x, wg_y, wg_z));
    }
}

// ── Full model: Encoder → Manipulator → Decoder ─────────────────────────────

pub struct MotionMagModel {
    // Encoder
    pub enc_conv1: ConvLayer,
    pub enc_conv2: ConvLayer,
    pub enc_conv3: ConvLayer,
    pub enc_texture: ConvLayer,

    pub manip_pipeline: ComputePipeline,

    // Decoder
    pub dec_conv1: ConvLayer,
    pub dec_conv2: ConvLayer,
    pub dec_conv3: ConvLayer,

    pub upsample_pipeline: ComputePipeline,
    pub rgba_to_chw_pipeline: ComputePipeline,
    pub chw_to_rgba_pipeline: ComputePipeline,

    // Dimensions
    pub width: u32,
    pub height: u32,
    pub latent_width: u32,
    pub latent_height: u32,

    // Pre-allocated intermediate buffers (reused every frame)
    buf_frame_a: Buffer,
    buf_frame_b: Buffer,
    buf_chw_a: Buffer,
    buf_chw_b: Buffer,
    buf_enc1_a: Buffer,
    buf_enc2_a: Buffer,
    buf_shape_a: Buffer,
    buf_texture_a: Buffer,
    buf_enc1_b: Buffer,
    buf_enc2_b: Buffer,
    buf_shape_b: Buffer,
    buf_manip_out: Buffer,
    buf_dec1: Buffer,
    buf_upsampled: Buffer,
    buf_dec2: Buffer,
    buf_chw_out: Buffer,
    buf_rgba_out: Buffer,
    buf_staging: Buffer,
}

/// Pre-loaded weight data for all layers.
pub struct ModelWeights {
    pub enc_conv1_w: Vec<f32>, pub enc_conv1_b: Vec<f32>,
    pub enc_conv2_w: Vec<f32>, pub enc_conv2_b: Vec<f32>,
    pub enc_conv3_w: Vec<f32>, pub enc_conv3_b: Vec<f32>,
    pub enc_texture_w: Vec<f32>, pub enc_texture_b: Vec<f32>,
    pub dec_conv1_w: Vec<f32>, pub dec_conv1_b: Vec<f32>,
    pub dec_conv2_w: Vec<f32>, pub dec_conv2_b: Vec<f32>,
    pub dec_conv3_w: Vec<f32>, pub dec_conv3_b: Vec<f32>,
}

impl ModelWeights {
    /// Random Xavier-like initialization (for use when trained weights aren't available).
    pub fn random() -> Self {
        fn rand_w(n: usize) -> Vec<f32> {
            let scale = (2.0 / n as f32).sqrt() * 0.1;
            (0..n).map(|i| {
                let x = (i as f32 * 0.618033988) % 1.0;
                (x - 0.5) * scale
            }).collect()
        }
        Self {
            enc_conv1_w: rand_w(16 * 3 * 3 * 3), enc_conv1_b: vec![0.0; 16],
            enc_conv2_w: rand_w(32 * 16 * 3 * 3), enc_conv2_b: vec![0.0; 32],
            enc_conv3_w: rand_w(32 * 32 * 3 * 3), enc_conv3_b: vec![0.0; 32],
            enc_texture_w: rand_w(32 * 32 * 1 * 1), enc_texture_b: vec![0.0; 32],
            dec_conv1_w: rand_w(32 * 32 * 3 * 3), dec_conv1_b: vec![0.0; 32],
            dec_conv2_w: rand_w(16 * 32 * 3 * 3), dec_conv2_b: vec![0.0; 16],
            dec_conv3_w: rand_w(3 * 16 * 3 * 3), dec_conv3_b: vec![0.0; 3],
        }
    }
}

impl MotionMagModel {
    /// Build the model with the given weights.
    pub fn new(ctx: &GpuContext, w: u32, h: u32, weights: &ModelWeights) -> Self {
        let lw = w / 2;
        let lh = h / 2;

        let enc_conv1 = ConvLayer::new(
            ctx, "enc_conv1",
            ConvParams {
                in_channels: 3, out_channels: 16, kernel_size: 3,
                stride: 1, padding: 1, width: w, height: h, use_relu: 1,
            },
            &weights.enc_conv1_w, &weights.enc_conv1_b,
        );

        let enc_conv2 = ConvLayer::new(
            ctx, "enc_conv2",
            ConvParams {
                in_channels: 16, out_channels: 32, kernel_size: 3,
                stride: 2, padding: 1, width: w, height: h, use_relu: 1,
            },
            &weights.enc_conv2_w, &weights.enc_conv2_b,
        );

        let enc_conv3 = ConvLayer::new(
            ctx, "enc_conv3",
            ConvParams {
                in_channels: 32, out_channels: 32, kernel_size: 3,
                stride: 1, padding: 1, width: lw, height: lh, use_relu: 1,
            },
            &weights.enc_conv3_w, &weights.enc_conv3_b,
        );

        let enc_texture = ConvLayer::new(
            ctx, "enc_texture",
            ConvParams {
                in_channels: 32, out_channels: 32, kernel_size: 1,
                stride: 1, padding: 0, width: lw, height: lh, use_relu: 0,
            },
            &weights.enc_texture_w, &weights.enc_texture_b,
        );

        let manip_pipeline = build_manipulator_pipeline(ctx);

        let dec_conv1 = ConvLayer::new(
            ctx, "dec_conv1",
            ConvParams {
                in_channels: 32, out_channels: 32, kernel_size: 3,
                stride: 1, padding: 1, width: lw, height: lh, use_relu: 1,
            },
            &weights.dec_conv1_w, &weights.dec_conv1_b,
        );

        let dec_conv2 = ConvLayer::new(
            ctx, "dec_conv2",
            ConvParams {
                in_channels: 32, out_channels: 16, kernel_size: 3,
                stride: 1, padding: 1, width: w, height: h, use_relu: 1,
            },
            &weights.dec_conv2_w, &weights.dec_conv2_b,
        );

        let dec_conv3 = ConvLayer::new(
            ctx, "dec_conv3",
            ConvParams {
                in_channels: 16, out_channels: 3, kernel_size: 3,
                stride: 1, padding: 1, width: w, height: h, use_relu: 0,
            },
            &weights.dec_conv3_w, &weights.dec_conv3_b,
        );

        let upsample_pipeline = build_upsample_pipeline(ctx);
        let rgba_to_chw_pipeline = build_frame_pipeline(ctx, &ctx.rgba_to_chw_module);
        let chw_to_rgba_pipeline = build_frame_pipeline(ctx, &ctx.chw_to_rgba_module);

        let pixels = (w * h) as u64;
        let latent_pixels = (lw * lh) as u64;
        let s = BufferUsages::STORAGE;
        let sc = BufferUsages::STORAGE | BufferUsages::COPY_SRC;

        Self {
            enc_conv1, enc_conv2, enc_conv3, enc_texture,
            manip_pipeline,
            dec_conv1, dec_conv2, dec_conv3,
            upsample_pipeline,
            rgba_to_chw_pipeline,
            chw_to_rgba_pipeline,
            width: w,
            height: h,
            latent_width: lw,
            latent_height: lh,

            buf_frame_a: ctx.create_buffer("frame_a", pixels * 4, sc),
            buf_frame_b: ctx.create_buffer("frame_b", pixels * 4, sc),
            buf_chw_a: ctx.create_buffer("chw_a", 3 * pixels * 4, s),
            buf_chw_b: ctx.create_buffer("chw_b", 3 * pixels * 4, s),
            buf_enc1_a: ctx.create_buffer("enc1_a", 16 * pixels * 4, s),
            buf_enc2_a: ctx.create_buffer("enc2_a", 32 * latent_pixels * 4, s),
            buf_shape_a: ctx.create_buffer("shape_a", 32 * latent_pixels * 4, s),
            buf_texture_a: ctx.create_buffer("texture_a", 32 * latent_pixels * 4, s),
            buf_enc1_b: ctx.create_buffer("enc1_b", 16 * pixels * 4, s),
            buf_enc2_b: ctx.create_buffer("enc2_b", 32 * latent_pixels * 4, s),
            buf_shape_b: ctx.create_buffer("shape_b", 32 * latent_pixels * 4, s),
            buf_manip_out: ctx.create_buffer("manip_out", 32 * latent_pixels * 4, s),
            buf_dec1: ctx.create_buffer("dec1", 32 * latent_pixels * 4, s),
            buf_upsampled: ctx.create_buffer("upsampled", 32 * pixels * 4, s),
            buf_dec2: ctx.create_buffer("dec2", 16 * pixels * 4, s),
            buf_chw_out: ctx.create_buffer("chw_out", 3 * pixels * 4, s),
            buf_rgba_out: ctx.create_buffer("rgba_out", pixels * 4, sc),
            buf_staging: ctx.create_buffer("staging", pixels * 4,
                BufferUsages::MAP_READ | BufferUsages::COPY_DST),
        }
    }

    /// Run the full magnification pipeline on two RGBA frames.
    /// Uses pre-allocated buffers to avoid GPU memory leaks.
    pub fn magnify(
        &self,
        ctx: &GpuContext,
        frame_a_rgba: &[u32],
        frame_b_rgba: &[u32],
        alpha: f32,
    ) {
        let w = self.width;
        let h = self.height;
        let lw = self.latent_width;
        let lh = self.latent_height;

        // Upload RGBA frames into pre-allocated buffers
        ctx.queue.write_buffer(&self.buf_frame_a, 0, bytemuck::cast_slice(frame_a_rgba));
        ctx.queue.write_buffer(&self.buf_frame_b, 0, bytemuck::cast_slice(frame_b_rgba));

        // Convert RGBA → CHW float
        self.run_rgba_to_chw(ctx, &self.buf_frame_a, &self.buf_chw_a, w, h);
        self.run_rgba_to_chw(ctx, &self.buf_frame_b, &self.buf_chw_b, w, h);

        // Encode frame A
        self.enc_conv1.run(ctx, &self.buf_chw_a, &self.buf_enc1_a);
        self.enc_conv2.run(ctx, &self.buf_enc1_a, &self.buf_enc2_a);
        self.enc_conv3.run(ctx, &self.buf_enc2_a, &self.buf_shape_a);
        self.enc_texture.run(ctx, &self.buf_enc2_a, &self.buf_texture_a);

        // Encode frame B
        self.enc_conv1.run(ctx, &self.buf_chw_b, &self.buf_enc1_b);
        self.enc_conv2.run(ctx, &self.buf_enc1_b, &self.buf_enc2_b);
        self.enc_conv3.run(ctx, &self.buf_enc2_b, &self.buf_shape_b);

        // Manipulate: amplify motion
        self.run_manipulator(ctx, &self.buf_shape_a, &self.buf_shape_b,
            &self.buf_texture_a, &self.buf_manip_out, 32, lh, lw, alpha);

        // Decode
        self.dec_conv1.run(ctx, &self.buf_manip_out, &self.buf_dec1);
        self.run_upsample(ctx, &self.buf_dec1, &self.buf_upsampled, 32, lh, lw, h, w);
        self.dec_conv2.run(ctx, &self.buf_upsampled, &self.buf_dec2);
        self.dec_conv3.run(ctx, &self.buf_dec2, &self.buf_chw_out);

        // Convert CHW → RGBA
        self.run_chw_to_rgba(ctx, &self.buf_chw_out, &self.buf_rgba_out, w, h);

        // Copy to staging for readback
        let mut encoder = ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("readback") },
        );
        encoder.copy_buffer_to_buffer(
            &self.buf_rgba_out, 0,
            &self.buf_staging, 0,
            (w * h * 4) as u64,
        );
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    fn run_rgba_to_chw(&self, ctx: &GpuContext, input: &Buffer, output: &Buffer, w: u32, h: u32) {
        let params = FrameParams { width: w, height: h, _pad0: 0, _pad1: 0 };
        let param_buf = ctx.create_uniform("frame_params", &params);
        let bg = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("rgba_to_chw_bg"),
            layout: &self.rgba_to_chw_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: param_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
            ],
        });
        ctx.dispatch(&self.rgba_to_chw_pipeline, &bg, (GpuContext::div_ceil(w, 8), GpuContext::div_ceil(h, 8), 1));
    }

    fn run_chw_to_rgba(&self, ctx: &GpuContext, input: &Buffer, output: &Buffer, w: u32, h: u32) {
        let params = FrameParams { width: w, height: h, _pad0: 0, _pad1: 0 };
        let param_buf = ctx.create_uniform("frame_params", &params);
        let bg = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("chw_to_rgba_bg"),
            layout: &self.chw_to_rgba_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: param_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
            ],
        });
        ctx.dispatch(&self.chw_to_rgba_pipeline, &bg, (GpuContext::div_ceil(w, 8), GpuContext::div_ceil(h, 8), 1));
    }

    fn run_manipulator(
        &self, ctx: &GpuContext,
        shape_a: &Buffer, shape_b: &Buffer, texture: &Buffer, output: &Buffer,
        channels: u32, h: u32, w: u32, alpha: f32,
    ) {
        let params = ManipParams { channels, height: h, width: w, alpha };
        let param_buf = ctx.create_uniform("manip_params", &params);
        let bg = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("manip_bg"),
            layout: &self.manip_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: param_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: shape_a.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: shape_b.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: texture.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: output.as_entire_binding() },
            ],
        });
        ctx.dispatch(&self.manip_pipeline, &bg,
            (GpuContext::div_ceil(w, 8), GpuContext::div_ceil(h, 8), channels));
    }

    fn run_upsample(
        &self, ctx: &GpuContext,
        input: &Buffer, output: &Buffer,
        channels: u32, in_h: u32, in_w: u32, out_h: u32, out_w: u32,
    ) {
        let params = UpsampleParams {
            channels, in_height: in_h, in_width: in_w,
            out_height: out_h, out_width: out_w, _pad: 0,
        };
        let param_buf = ctx.create_uniform("upsample_params", &params);
        let bg = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("upsample_bg"),
            layout: &self.upsample_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: param_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: output.as_entire_binding() },
            ],
        });
        ctx.dispatch(&self.upsample_pipeline, &bg,
            (GpuContext::div_ceil(out_w, 8), GpuContext::div_ceil(out_h, 8), channels));
    }
}

// ── Pipeline builders ────────────────────────────────────────────────────────

fn build_manipulator_pipeline(ctx: &GpuContext) -> ComputePipeline {
    ctx.device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("manipulator"),
        layout: None,
        module: &ctx.manipulator_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn build_upsample_pipeline(ctx: &GpuContext) -> ComputePipeline {
    ctx.device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("upsample"),
        layout: None,
        module: &ctx.upsample_module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

fn build_frame_pipeline(ctx: &GpuContext, module: &ShaderModule) -> ComputePipeline {
    ctx.device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("frame_io"),
        layout: None,
        module,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn bgl_entry(
    binding: u32,
    visibility: ShaderStages,
    is_uniform: bool,
    read_only: bool,
) -> BindGroupLayoutEntry {
    if is_uniform {
        BindGroupLayoutEntry {
            binding,
            visibility,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    } else {
        BindGroupLayoutEntry {
            binding,
            visibility,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }
    }
}


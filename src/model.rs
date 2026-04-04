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
    pub params: ConvParams,
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
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });

        let pipeline = ctx.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some(label),
            layout: Some(&pipeline_layout),
            module: &ctx.conv2d_module,
            entry_point: "main",
        });

        Self {
            params,
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
    // Encoder: single-branch (Ha et al. finding: one branch suffices)
    pub enc_conv1: ConvLayer,   // 3 → 16, k=3, s=1, p=1
    pub enc_conv2: ConvLayer,   // 16 → 32, k=3, s=2, p=1  (downsample)
    pub enc_conv3: ConvLayer,   // 32 → 32, k=3, s=1, p=1

    // Encoder produces: shape_rep (32ch) and texture_rep (32ch)
    // Ha et al.: single linear layer for texture branch
    pub enc_texture: ConvLayer, // 32 → 32, k=1, s=1, p=0

    // Manipulator pipeline (built separately, uses WGSL manipulator shader)
    pub manip_pipeline: ComputePipeline,

    // Decoder: reduced-resolution path (Ha et al. finding #1)
    pub dec_conv1: ConvLayer,   // 32 → 32, k=3, s=1, p=1
    // upsample 2×
    pub dec_conv2: ConvLayer,   // 32 → 16, k=3, s=1, p=1
    pub dec_conv3: ConvLayer,   // 16 → 3, k=3, s=1, p=1 (no relu)

    pub upsample_pipeline: ComputePipeline,

    // Frame I/O pipelines
    pub rgba_to_chw_pipeline: ComputePipeline,
    pub chw_to_rgba_pipeline: ComputePipeline,

    // Dimensions
    pub width: u32,
    pub height: u32,
    pub latent_width: u32,
    pub latent_height: u32,
}

impl MotionMagModel {
    /// Build the model with random weights (replace with loaded weights for real use).
    /// `w` and `h` are input frame dimensions.
    pub fn new(ctx: &GpuContext, w: u32, h: u32) -> Self {
        let lw = w / 2;  // latent spatial dims after stride-2 conv
        let lh = h / 2;

        // ── Encoder ──
        let enc_conv1 = ConvLayer::new(
            ctx, "enc_conv1",
            ConvParams {
                in_channels: 3, out_channels: 16, kernel_size: 3,
                stride: 1, padding: 1, width: w, height: h, use_relu: 1,
            },
            &random_weights(16 * 3 * 3 * 3),
            &vec![0.0; 16],
        );

        let enc_conv2 = ConvLayer::new(
            ctx, "enc_conv2",
            ConvParams {
                in_channels: 16, out_channels: 32, kernel_size: 3,
                stride: 2, padding: 1, width: w, height: h, use_relu: 1,
            },
            &random_weights(32 * 16 * 3 * 3),
            &vec![0.0; 32],
        );

        let enc_conv3 = ConvLayer::new(
            ctx, "enc_conv3",
            ConvParams {
                in_channels: 32, out_channels: 32, kernel_size: 3,
                stride: 1, padding: 1, width: lw, height: lh, use_relu: 1,
            },
            &random_weights(32 * 32 * 3 * 3),
            &vec![0.0; 32],
        );

        // Texture branch: 1×1 conv (effectively a per-pixel linear layer)
        let enc_texture = ConvLayer::new(
            ctx, "enc_texture",
            ConvParams {
                in_channels: 32, out_channels: 32, kernel_size: 1,
                stride: 1, padding: 0, width: lw, height: lh, use_relu: 0,
            },
            &random_weights(32 * 32 * 1 * 1),
            &vec![0.0; 32],
        );

        // ── Manipulator pipeline ──
        let manip_pipeline = build_manipulator_pipeline(ctx);

        // ── Decoder ──
        let dec_conv1 = ConvLayer::new(
            ctx, "dec_conv1",
            ConvParams {
                in_channels: 32, out_channels: 32, kernel_size: 3,
                stride: 1, padding: 1, width: lw, height: lh, use_relu: 1,
            },
            &random_weights(32 * 32 * 3 * 3),
            &vec![0.0; 32],
        );

        // After upsample: back to full resolution
        let dec_conv2 = ConvLayer::new(
            ctx, "dec_conv2",
            ConvParams {
                in_channels: 32, out_channels: 16, kernel_size: 3,
                stride: 1, padding: 1, width: w, height: h, use_relu: 1,
            },
            &random_weights(16 * 32 * 3 * 3),
            &vec![0.0; 16],
        );

        let dec_conv3 = ConvLayer::new(
            ctx, "dec_conv3",
            ConvParams {
                in_channels: 16, out_channels: 3, kernel_size: 3,
                stride: 1, padding: 1, width: w, height: h, use_relu: 0,  // final: no relu
            },
            &random_weights(3 * 16 * 3 * 3),
            &vec![0.0; 3],
        );

        let upsample_pipeline = build_upsample_pipeline(ctx);

        let rgba_to_chw_pipeline = build_frame_pipeline(ctx, "rgba_to_chw");
        let chw_to_rgba_pipeline = build_frame_pipeline(ctx, "chw_to_rgba");

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
        }
    }

    /// Run the full magnification pipeline on two RGBA frames.
    /// Returns magnified RGBA pixel data.
    pub fn magnify(
        &self,
        ctx: &GpuContext,
        frame_a_rgba: &[u32],
        frame_b_rgba: &[u32],
        alpha: f32,
    ) -> Buffer {
        let w = self.width;
        let h = self.height;
        let lw = self.latent_width;
        let lh = self.latent_height;
        let pixels = (w * h) as usize;
        let latent_pixels = (lw * lh) as usize;

        // Upload RGBA frames
        let frame_a_buf = ctx.create_buffer_init("frame_a", bytemuck::cast_slice(frame_a_rgba), BufferUsages::STORAGE);
        let frame_b_buf = ctx.create_buffer_init("frame_b", bytemuck::cast_slice(frame_b_rgba), BufferUsages::STORAGE);

        // Convert RGBA → CHW float
        let chw_a = ctx.create_buffer("chw_a", (3 * pixels * 4) as u64, BufferUsages::STORAGE);
        let chw_b = ctx.create_buffer("chw_b", (3 * pixels * 4) as u64, BufferUsages::STORAGE);
        self.run_rgba_to_chw(ctx, &frame_a_buf, &chw_a, w, h);
        self.run_rgba_to_chw(ctx, &frame_b_buf, &chw_b, w, h);

        // Encode both frames
        let enc1_a = ctx.create_buffer("enc1_a", (16 * pixels * 4) as u64, BufferUsages::STORAGE);
        let enc2_a = ctx.create_buffer("enc2_a", (32 * latent_pixels * 4) as u64, BufferUsages::STORAGE);
        let shape_a = ctx.create_buffer("shape_a", (32 * latent_pixels * 4) as u64, BufferUsages::STORAGE);
        let texture_a = ctx.create_buffer("texture_a", (32 * latent_pixels * 4) as u64, BufferUsages::STORAGE);

        self.enc_conv1.run(ctx, &chw_a, &enc1_a);
        self.enc_conv2.run(ctx, &enc1_a, &enc2_a);
        self.enc_conv3.run(ctx, &enc2_a, &shape_a);
        self.enc_texture.run(ctx, &enc2_a, &texture_a);

        let enc1_b = ctx.create_buffer("enc1_b", (16 * pixels * 4) as u64, BufferUsages::STORAGE);
        let enc2_b = ctx.create_buffer("enc2_b", (32 * latent_pixels * 4) as u64, BufferUsages::STORAGE);
        let shape_b = ctx.create_buffer("shape_b", (32 * latent_pixels * 4) as u64, BufferUsages::STORAGE);

        self.enc_conv1.run(ctx, &chw_b, &enc1_b);
        self.enc_conv2.run(ctx, &enc1_b, &enc2_b);
        self.enc_conv3.run(ctx, &enc2_b, &shape_b);

        // Manipulate: amplify motion
        let manip_out = ctx.create_buffer("manip_out", (32 * latent_pixels * 4) as u64, BufferUsages::STORAGE);
        self.run_manipulator(ctx, &shape_a, &shape_b, &texture_a, &manip_out, 32, lh, lw, alpha);

        // Decode
        let dec1_out = ctx.create_buffer("dec1", (32 * latent_pixels * 4) as u64, BufferUsages::STORAGE);
        self.dec_conv1.run(ctx, &manip_out, &dec1_out);

        // Upsample 2× back to full resolution
        let upsampled = ctx.create_buffer("upsampled", (32 * pixels * 4) as u64, BufferUsages::STORAGE);
        self.run_upsample(ctx, &dec1_out, &upsampled, 32, lh, lw, h, w);

        let dec2_out = ctx.create_buffer("dec2", (16 * pixels * 4) as u64, BufferUsages::STORAGE);
        self.dec_conv2.run(ctx, &upsampled, &dec2_out);

        let chw_out = ctx.create_buffer("chw_out", (3 * pixels * 4) as u64, BufferUsages::STORAGE);
        self.dec_conv3.run(ctx, &dec2_out, &chw_out);

        // Convert CHW → RGBA
        let rgba_out = ctx.create_buffer("rgba_out", (pixels * 4) as u64,
            BufferUsages::STORAGE | BufferUsages::COPY_SRC);
        self.run_chw_to_rgba(ctx, &chw_out, &rgba_out, w, h);

        rgba_out
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
        layout: None,  // auto layout
        module: &ctx.manipulator_module,
        entry_point: "main",
    })
}

fn build_upsample_pipeline(ctx: &GpuContext) -> ComputePipeline {
    ctx.device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some("upsample"),
        layout: None,
        module: &ctx.upsample_module,
        entry_point: "main",
    })
}

fn build_frame_pipeline(ctx: &GpuContext, entry: &str) -> ComputePipeline {
    ctx.device.create_compute_pipeline(&ComputePipelineDescriptor {
        label: Some(entry),
        layout: None,
        module: &ctx.frame_io_module,
        entry_point: entry,
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

/// Placeholder: generate random weights for demo. Replace with weight loading.
fn random_weights(n: usize) -> Vec<f32> {
    // Xavier-ish init: small random values
    let scale = (2.0 / n as f32).sqrt() * 0.1;
    (0..n).map(|i| {
        let x = (i as f32 * 0.618033988) % 1.0;
        (x - 0.5) * scale
    }).collect()
}

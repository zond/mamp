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

    /// Create a pre-built bind group for a specific input/output buffer pair.
    /// This avoids per-frame bind group allocation for conv layers whose
    /// buffers are fixed across frames.
    pub fn create_bind_group(&self, ctx: &GpuContext, label: &str, input: &Buffer, output: &Buffer) -> BindGroup {
        ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some(label),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: self.param_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: input.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: self.weight_buf.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: self.bias_buf.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: output.as_entire_binding() },
            ],
        })
    }

    /// Record this conv layer's dispatch into a new compute pass on the encoder.
    /// Each dispatch gets its own pass to ensure proper memory barriers between
    /// dependent shader invocations.
    pub fn record(
        &self,
        encoder: &mut CommandEncoder,
        bind_group: &BindGroup,
    ) {
        let wg_x = GpuContext::div_ceil(self.out_width, 8);
        let wg_y = GpuContext::div_ceil(self.out_height, 8);
        let wg_z = self.out_channels;

        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("conv_pass"),
            timestamp_writes: None,
        });
        GpuContext::record_dispatch(&mut pass, &self.pipeline, bind_group, (wg_x, wg_y, wg_z));
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
    pub buf_staging: [Buffer; 2],

    // Pre-allocated uniform buffers (avoid per-frame GPU allocation)
    frame_param_buf: Buffer,       // shared FrameParams uniform (width/height are fixed)
    manip_param_buf: Buffer,       // ManipParams uniform (alpha updated per frame via write_buffer)
    upsample_param_buf: Buffer,    // UpsampleParams uniform (fixed)

    // Pre-allocated bind groups for every dispatch (avoid per-frame allocation).
    // Buffers are fixed across frames, so bind groups can be created once at init.
    rgba_to_chw_a_bg: BindGroup,   // frame_a -> chw_a
    rgba_to_chw_b_bg: BindGroup,   // frame_b -> chw_b
    chw_to_rgba_bg: BindGroup,     // chw_out -> rgba_out
    manip_bg: BindGroup,           // shape_a, shape_b, texture_a -> manip_out
    upsample_bg: BindGroup,        // dec1 -> upsampled
    enc_conv1_a_bg: BindGroup,     // chw_a -> enc1_a
    enc_conv2_a_bg: BindGroup,     // enc1_a -> enc2_a
    enc_conv3_a_bg: BindGroup,     // enc2_a -> shape_a
    enc_texture_a_bg: BindGroup,   // enc2_a -> texture_a
    enc_conv1_b_bg: BindGroup,     // chw_b -> enc1_b
    enc_conv2_b_bg: BindGroup,     // enc1_b -> enc2_b
    enc_conv3_b_bg: BindGroup,     // enc2_b -> shape_b
    dec_conv1_bg: BindGroup,       // manip_out -> dec1
    dec_conv2_bg: BindGroup,       // upsampled -> dec2
    dec_conv3_bg: BindGroup,       // dec2 -> chw_out
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
    /// All intermediate buffers, uniform buffers, and bind groups are pre-allocated
    /// so that `magnify()` performs zero GPU allocations per frame.
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
        let scd = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;

        // Allocate all intermediate buffers
        let buf_frame_a = ctx.create_buffer("frame_a", pixels * 4, scd);
        let buf_frame_b = ctx.create_buffer("frame_b", pixels * 4, scd);
        let buf_chw_a = ctx.create_buffer("chw_a", 3 * pixels * 4, s);
        let buf_chw_b = ctx.create_buffer("chw_b", 3 * pixels * 4, s);
        let buf_enc1_a = ctx.create_buffer("enc1_a", 16 * pixels * 4, s);
        let buf_enc2_a = ctx.create_buffer("enc2_a", 32 * latent_pixels * 4, s);
        let buf_shape_a = ctx.create_buffer("shape_a", 32 * latent_pixels * 4, s);
        let buf_texture_a = ctx.create_buffer("texture_a", 32 * latent_pixels * 4, s);
        let buf_enc1_b = ctx.create_buffer("enc1_b", 16 * pixels * 4, s);
        let buf_enc2_b = ctx.create_buffer("enc2_b", 32 * latent_pixels * 4, s);
        let buf_shape_b = ctx.create_buffer("shape_b", 32 * latent_pixels * 4, s);
        let buf_manip_out = ctx.create_buffer("manip_out", 32 * latent_pixels * 4, s);
        let buf_dec1 = ctx.create_buffer("dec1", 32 * latent_pixels * 4, s);
        let buf_upsampled = ctx.create_buffer("upsampled", 32 * pixels * 4, s);
        let buf_dec2 = ctx.create_buffer("dec2", 16 * pixels * 4, s);
        let buf_chw_out = ctx.create_buffer("chw_out", 3 * pixels * 4, s);
        let buf_rgba_out = ctx.create_buffer("rgba_out", pixels * 4, sc);
        let buf_staging = [
            ctx.create_buffer("staging_0", pixels * 4, BufferUsages::MAP_READ | BufferUsages::COPY_DST),
            ctx.create_buffer("staging_1", pixels * 4, BufferUsages::MAP_READ | BufferUsages::COPY_DST),
        ];

        // Pre-allocate uniform buffers
        let frame_params = FrameParams { width: w, height: h, _pad0: 0, _pad1: 0 };
        let frame_param_buf = ctx.create_uniform("frame_params", &frame_params);

        let manip_params = ManipParams { channels: 32, height: lh, width: lw, alpha: 0.0 };
        let manip_param_buf = ctx.create_uniform("manip_params", &manip_params);

        let upsample_params = UpsampleParams {
            channels: 32, in_height: lh, in_width: lw,
            out_height: h, out_width: w, _pad: 0,
        };
        let upsample_param_buf = ctx.create_uniform("upsample_params", &upsample_params);

        // Pre-allocate all bind groups (buffers are fixed, so these never change)
        let rgba_to_chw_a_bg = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("rgba_to_chw_a_bg"),
            layout: &rgba_to_chw_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: frame_param_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: buf_frame_a.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: buf_chw_a.as_entire_binding() },
            ],
        });

        let rgba_to_chw_b_bg = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("rgba_to_chw_b_bg"),
            layout: &rgba_to_chw_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: frame_param_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: buf_frame_b.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: buf_chw_b.as_entire_binding() },
            ],
        });

        let chw_to_rgba_bg = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("chw_to_rgba_bg"),
            layout: &chw_to_rgba_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: frame_param_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: buf_chw_out.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: buf_rgba_out.as_entire_binding() },
            ],
        });

        let manip_bg = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("manip_bg"),
            layout: &manip_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: manip_param_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: buf_shape_a.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: buf_shape_b.as_entire_binding() },
                BindGroupEntry { binding: 3, resource: buf_texture_a.as_entire_binding() },
                BindGroupEntry { binding: 4, resource: buf_manip_out.as_entire_binding() },
            ],
        });

        let upsample_bg = ctx.device.create_bind_group(&BindGroupDescriptor {
            label: Some("upsample_bg"),
            layout: &upsample_pipeline.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: upsample_param_buf.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: buf_dec1.as_entire_binding() },
                BindGroupEntry { binding: 2, resource: buf_upsampled.as_entire_binding() },
            ],
        });

        // Conv layer bind groups (encoder path A)
        let enc_conv1_a_bg = enc_conv1.create_bind_group(ctx, "enc_conv1_a_bg", &buf_chw_a, &buf_enc1_a);
        let enc_conv2_a_bg = enc_conv2.create_bind_group(ctx, "enc_conv2_a_bg", &buf_enc1_a, &buf_enc2_a);
        let enc_conv3_a_bg = enc_conv3.create_bind_group(ctx, "enc_conv3_a_bg", &buf_enc2_a, &buf_shape_a);
        let enc_texture_a_bg = enc_texture.create_bind_group(ctx, "enc_texture_a_bg", &buf_enc2_a, &buf_texture_a);

        // Conv layer bind groups (encoder path B)
        let enc_conv1_b_bg = enc_conv1.create_bind_group(ctx, "enc_conv1_b_bg", &buf_chw_b, &buf_enc1_b);
        let enc_conv2_b_bg = enc_conv2.create_bind_group(ctx, "enc_conv2_b_bg", &buf_enc1_b, &buf_enc2_b);
        let enc_conv3_b_bg = enc_conv3.create_bind_group(ctx, "enc_conv3_b_bg", &buf_enc2_b, &buf_shape_b);

        // Conv layer bind groups (decoder)
        let dec_conv1_bg = dec_conv1.create_bind_group(ctx, "dec_conv1_bg", &buf_manip_out, &buf_dec1);
        let dec_conv2_bg = dec_conv2.create_bind_group(ctx, "dec_conv2_bg", &buf_upsampled, &buf_dec2);
        let dec_conv3_bg = dec_conv3.create_bind_group(ctx, "dec_conv3_bg", &buf_dec2, &buf_chw_out);

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

            buf_frame_a, buf_frame_b,
            buf_chw_a, buf_chw_b,
            buf_enc1_a, buf_enc2_a, buf_shape_a, buf_texture_a,
            buf_enc1_b, buf_enc2_b, buf_shape_b,
            buf_manip_out, buf_dec1, buf_upsampled, buf_dec2,
            buf_chw_out, buf_rgba_out, buf_staging,

            frame_param_buf, manip_param_buf, upsample_param_buf,

            rgba_to_chw_a_bg, rgba_to_chw_b_bg, chw_to_rgba_bg,
            manip_bg, upsample_bg,
            enc_conv1_a_bg, enc_conv2_a_bg, enc_conv3_a_bg, enc_texture_a_bg,
            enc_conv1_b_bg, enc_conv2_b_bg, enc_conv3_b_bg,
            dec_conv1_bg, dec_conv2_bg, dec_conv3_bg,
        }
    }

    /// Run the full magnification pipeline on two RGBA frames.
    ///
    /// All ~15 compute dispatches and the staging buffer copy are recorded into
    /// a single command encoder and submitted with one `queue.submit()` call.
    /// This eliminates per-dispatch submission overhead, which is the primary
    /// source of jank on mobile GPUs with high driver-side submit cost.
    pub fn magnify(
        &self,
        ctx: &GpuContext,
        frame_a_rgba: &[u32],
        frame_b_rgba: &[u32],
        alpha: f32,
        staging_idx: usize,
    ) {
        let w = self.width;
        let h = self.height;
        let lw = self.latent_width;
        let lh = self.latent_height;

        // Upload RGBA frames and update the alpha uniform via write_buffer.
        // These are queued internally and will execute before our command buffer.
        ctx.queue.write_buffer(&self.buf_frame_a, 0, bytemuck::cast_slice(frame_a_rgba));
        ctx.queue.write_buffer(&self.buf_frame_b, 0, bytemuck::cast_slice(frame_b_rgba));

        let manip_params = ManipParams { channels: 32, height: lh, width: lw, alpha };
        ctx.queue.write_buffer(&self.manip_param_buf, 0, bytemuck::bytes_of(&manip_params));

        // Single command encoder for the entire frame
        let mut encoder = ctx.device.create_command_encoder(
            &CommandEncoderDescriptor { label: Some("magnify") },
        );

        // Convert RGBA -> CHW float (frame A and B)
        self.record_frame_dispatch(&mut encoder, "rgba_to_chw_a",
            &self.rgba_to_chw_pipeline, &self.rgba_to_chw_a_bg, w, h);
        self.record_frame_dispatch(&mut encoder, "rgba_to_chw_b",
            &self.rgba_to_chw_pipeline, &self.rgba_to_chw_b_bg, w, h);

        // Encode frame A: conv1 -> conv2 -> conv3 (shape) + texture
        self.enc_conv1.record(&mut encoder, &self.enc_conv1_a_bg);
        self.enc_conv2.record(&mut encoder, &self.enc_conv2_a_bg);
        self.enc_conv3.record(&mut encoder, &self.enc_conv3_a_bg);
        self.enc_texture.record(&mut encoder, &self.enc_texture_a_bg);

        // Encode frame B: conv1 -> conv2 -> conv3 (shape)
        self.enc_conv1.record(&mut encoder, &self.enc_conv1_b_bg);
        self.enc_conv2.record(&mut encoder, &self.enc_conv2_b_bg);
        self.enc_conv3.record(&mut encoder, &self.enc_conv3_b_bg);

        // Manipulate: amplify motion difference
        self.record_manip_dispatch(&mut encoder, lw, lh);

        // Decode: conv1 -> upsample -> conv2 -> conv3
        self.dec_conv1.record(&mut encoder, &self.dec_conv1_bg);
        self.record_upsample_dispatch(&mut encoder, w, h);
        self.dec_conv2.record(&mut encoder, &self.dec_conv2_bg);
        self.dec_conv3.record(&mut encoder, &self.dec_conv3_bg);

        // Convert CHW -> RGBA
        self.record_frame_dispatch(&mut encoder, "chw_to_rgba",
            &self.chw_to_rgba_pipeline, &self.chw_to_rgba_bg, w, h);

        // Copy result to staging buffer for CPU readback
        encoder.copy_buffer_to_buffer(
            &self.buf_rgba_out, 0,
            &self.buf_staging[staging_idx], 0,
            (w * h * 4) as u64,
        );

        // Single submit for the entire frame
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Run magnification without staging copy (for when readback is handled separately).
    pub fn magnify_no_staging(
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

        ctx.queue.write_buffer(&self.buf_frame_a, 0, bytemuck::cast_slice(frame_a_rgba));
        ctx.queue.write_buffer(&self.buf_frame_b, 0, bytemuck::cast_slice(frame_b_rgba));

        let manip_params = ManipParams { channels: 32, height: lh, width: lw, alpha };
        ctx.queue.write_buffer(&self.manip_param_buf, 0, bytemuck::bytes_of(&manip_params));

        let mut encoder = ctx.device.create_command_encoder(
            &CommandEncoderDescriptor { label: Some("magnify") },
        );

        self.record_frame_dispatch(&mut encoder, "rgba_to_chw_a",
            &self.rgba_to_chw_pipeline, &self.rgba_to_chw_a_bg, w, h);
        self.record_frame_dispatch(&mut encoder, "rgba_to_chw_b",
            &self.rgba_to_chw_pipeline, &self.rgba_to_chw_b_bg, w, h);

        self.enc_conv1.record(&mut encoder, &self.enc_conv1_a_bg);
        self.enc_conv2.record(&mut encoder, &self.enc_conv2_a_bg);
        self.enc_conv3.record(&mut encoder, &self.enc_conv3_a_bg);
        self.enc_texture.record(&mut encoder, &self.enc_texture_a_bg);

        self.enc_conv1.record(&mut encoder, &self.enc_conv1_b_bg);
        self.enc_conv2.record(&mut encoder, &self.enc_conv2_b_bg);
        self.enc_conv3.record(&mut encoder, &self.enc_conv3_b_bg);

        self.record_manip_dispatch(&mut encoder, lw, lh);

        self.dec_conv1.record(&mut encoder, &self.dec_conv1_bg);
        self.record_upsample_dispatch(&mut encoder, w, h);
        self.dec_conv2.record(&mut encoder, &self.dec_conv2_bg);
        self.dec_conv3.record(&mut encoder, &self.dec_conv3_bg);

        self.record_frame_dispatch(&mut encoder, "chw_to_rgba",
            &self.chw_to_rgba_pipeline, &self.chw_to_rgba_bg, w, h);

        // No staging copy — result stays in buf_rgba_out on GPU
        ctx.queue.submit(std::iter::once(encoder.finish()));
    }

    /// Record a frame-format conversion dispatch (rgba_to_chw or chw_to_rgba).
    fn record_frame_dispatch(
        &self,
        encoder: &mut CommandEncoder,
        label: &str,
        pipeline: &ComputePipeline,
        bind_group: &BindGroup,
        w: u32,
        h: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some(label),
            timestamp_writes: None,
        });
        GpuContext::record_dispatch(
            &mut pass, pipeline, bind_group,
            (GpuContext::div_ceil(w, 64), GpuContext::div_ceil(h, 16), 1),
        );
    }

    /// Record the manipulator dispatch.
    fn record_manip_dispatch(&self, encoder: &mut CommandEncoder, w: u32, h: u32) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("manipulator"),
            timestamp_writes: None,
        });
        GpuContext::record_dispatch(
            &mut pass, &self.manip_pipeline, &self.manip_bg,
            (GpuContext::div_ceil(w, 8), GpuContext::div_ceil(h, 8), 8),
        );
    }

    /// Record the upsample dispatch.
    fn record_upsample_dispatch(&self, encoder: &mut CommandEncoder, out_w: u32, out_h: u32) {
        let mut pass = encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("upsample"),
            timestamp_writes: None,
        });
        GpuContext::record_dispatch(
            &mut pass, &self.upsample_pipeline, &self.upsample_bg,
            (GpuContext::div_ceil(out_w, 8), GpuContext::div_ceil(out_h, 8), 8),
        );
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


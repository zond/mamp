// steerable.rs — Complex Steerable Pyramid phase-based motion magnification
//
// Per frame (~92 GPU dispatches):
// 1. RGBA → Y,I,Q (pad)         2. FFT(Y)
// 3. For 12 bandpass sub-bands:  filter → IFFT → phase_amplify → FFT → filter_accum
// 4. Residuals: filter² → accum  5. IFFT(accum) → modified Y
// 6. YIQ → RGBA                  7. Blit to canvas

use crate::gpu::GpuContext;
use bytemuck::{Pod, Zeroable};
use wasm_bindgen::JsCast;
use wgpu::util::DeviceExt;
use wgpu::*;

const N_SCALES: u32 = 3;
const N_ORIENT: u32 = 4;
const N_BAND: usize = (N_SCALES * N_ORIENT) as usize; // 12

fn next_pow2(n: u32) -> u32 {
    1u32 << (32 - (n - 1).leading_zeros())
}

// ── Uniform structs ──

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
struct ColorParams { ow: u32, oh: u32, pw: u32, ph: u32 }

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
struct FftParams { n: u32, log2n: u32, num: u32, inv: u32, stride: u32, fft_stride: u32, _p0: u32, _p1: u32 }

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
struct FilterParams { w: u32, h: u32, norient: u32, oidx: u32, scale: u32, nscales: u32, ftype: u32, mode: u32 }

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
struct PhaseParams { w: u32, h: u32, amp: f32, al: f32, ah: f32, first: u32, _p0: u32, _p1: u32 }

#[repr(C)] #[derive(Copy, Clone, Pod, Zeroable)]
struct BlitParams { w: u32, h: u32 }

// ── Pipeline ──

#[allow(dead_code)]
pub struct SteerablePipeline {
    pub ow: u32, pub oh: u32, // original dimensions
    pub pw: u32, pub ph: u32, // padded (power of 2)

    // Compute pipelines
    pl_r2y: ComputePipeline,
    pl_y2r: ComputePipeline,
    pl_fft: ComputePipeline,
    pl_filt: ComputePipeline,
    pl_filt_acc: ComputePipeline,
    pl_phase: ComputePipeline,
    pl_blit: RenderPipeline,
    surface: Surface<'static>,

    // Buffers — frame I/O
    b_rgba_in: Buffer,
    b_rgba_out: Buffer,
    b_y: Buffer, b_i: Buffer, b_q: Buffer,
    b_zeros: Buffer,

    // Buffers — spectrum
    b_spec_re: Buffer, b_spec_im: Buffer,
    b_acc_re: Buffer, b_acc_im: Buffer,
    b_out_y: Buffer,
    b_discard_im: Buffer, // for discarding IFFT imaginary

    // Buffers — FFT intermediate (row↔col)
    b_tmp_re: Buffer, b_tmp_im: Buffer,

    // Buffers — per sub-band temporary (reused)
    b_ss_re: Buffer, b_ss_im: Buffer, // sub-band spectrum
    b_sp_re: Buffer, b_sp_im: Buffer, // sub-band spatial
    b_sm_re: Buffer, b_sm_im: Buffer, // sub-band modified
    b_ms_re: Buffer, b_ms_im: Buffer, // modified spectrum

    // Buffers — per sub-band persistent state (12 sets)
    prev_re: [Buffer; N_BAND],
    prev_im: [Buffer; N_BAND],
    lp_hi: [Buffer; N_BAND],
    lp_lo: [Buffer; N_BAND],

    // Pre-built uniforms (static)
    u_color: Buffer,
    u_blit: Buffer,
    u_phase: Buffer,
    // Filter uniforms (12 bandpass + 2 residuals)
    u_filt: [Buffer; N_BAND],
    u_filt_acc: [Buffer; N_BAND],
    u_res_hi: Buffer,
    u_res_lo: Buffer,
    // FFT uniforms (4 configs: row_fwd, col_fwd, row_inv, col_inv)
    u_fft_rf: Buffer, u_fft_cf: Buffer,
    u_fft_ri: Buffer, u_fft_ci: Buffer,

    first_frame: bool,
}

impl SteerablePipeline {
    pub fn new(ctx: &GpuContext, w: u32, h: u32, max_fft: u32) -> Self {
        let pw = next_pow2(w).min(max_fft);
        let ph = next_pow2(h).min(max_fft);
        let ppix = (pw * ph) as u64;
        let opix = (w * h) as u64;
        log::info!("Steerable: {}x{} pad {}x{}", w, h, pw, ph);

        let s = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        let mk = |name: &str, sz: u64| ctx.create_buffer(name, sz, s);

        let b_rgba_in = mk("si", opix * 4);
        let b_rgba_out = ctx.create_buffer("so", opix * 4, BufferUsages::STORAGE | BufferUsages::COPY_DST);
        let b_y = mk("sy", ppix * 4);
        let b_i = mk("si_c", ppix * 4);
        let b_q = mk("sq", ppix * 4);
        let b_zeros = mk("sz", ppix * 4);
        ctx.queue.write_buffer(&b_zeros, 0, &vec![0u8; (ppix * 4) as usize]);

        let b_spec_re = mk("sr", ppix * 4); let b_spec_im = mk("si2", ppix * 4);
        let b_acc_re = mk("ar", ppix * 4); let b_acc_im = mk("ai", ppix * 4);
        let b_out_y = mk("oy", ppix * 4);
        let b_discard_im = mk("di", ppix * 4);
        let b_tmp_re = mk("tr", ppix * 4); let b_tmp_im = mk("ti", ppix * 4);
        let b_ss_re = mk("ssr", ppix*4); let b_ss_im = mk("ssi", ppix*4);
        let b_sp_re = mk("spr", ppix*4); let b_sp_im = mk("spi", ppix*4);
        let b_sm_re = mk("smr", ppix*4); let b_sm_im = mk("smi", ppix*4);
        let b_ms_re = mk("msr", ppix*4); let b_ms_im = mk("msi", ppix*4);

        let prev_re = std::array::from_fn(|i| mk(&format!("pr{}", i), ppix*4));
        let prev_im = std::array::from_fn(|i| mk(&format!("pi{}", i), ppix*4));
        let lp_hi = std::array::from_fn(|i| mk(&format!("lh{}", i), ppix*4));
        let lp_lo = std::array::from_fn(|i| mk(&format!("ll{}", i), ppix*4));

        // ── Pipelines ──
        let mkc = |label: &str, src: &str| {
            let m = ctx.device.create_shader_module(ShaderModuleDescriptor {
                label: Some(label), source: ShaderSource::Wgsl(src.into()),
            });
            ctx.device.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label), layout: None, module: &m, entry_point: Some("main"),
                compilation_options: Default::default(), cache: None,
            })
        };

        let pl_r2y = mkc("r2y", include_str!("shaders/rgba_to_y.wgsl"));
        let pl_y2r = mkc("y2r", include_str!("shaders/yiq_to_rgba.wgsl"));
        let fft_src = include_str!("shaders/fft.wgsl").replace("/*SHARED_SIZE*/", &max_fft.to_string());
        let pl_fft = mkc("fft", &fft_src);
        let pl_filt = mkc("filt", include_str!("shaders/steerable_filters.wgsl"));
        let pl_filt_acc = mkc("facc", include_str!("shaders/filter_accumulate.wgsl"));
        let pl_phase = mkc("phase", include_str!("shaders/phase_amplify.wgsl"));

        // Surface + blit
        let canvas: web_sys::HtmlCanvasElement = web_sys::window().unwrap()
            .document().unwrap().get_element_by_id("output").unwrap().unchecked_into();
        canvas.set_width(w); canvas.set_height(h);
        let surface = ctx.instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas)).unwrap();
        let caps = surface.get_capabilities(&ctx.adapter);
        let fmt = caps.formats[0];
        surface.configure(&ctx.device, &SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT, format: fmt,
            width: w, height: h,
            present_mode: caps.present_modes[0],
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        });

        let blit_mod = ctx.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("blit"), source: ShaderSource::Wgsl(include_str!("shaders/blit.wgsl").into()),
        });
        let pl_blit = ctx.device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("blit"), layout: None,
            vertex: VertexState { module: &blit_mod, entry_point: Some("vs_main"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(FragmentState { module: &blit_mod, entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState { format: fmt, blend: None, write_mask: ColorWrites::ALL })],
                compilation_options: Default::default() }),
            primitive: PrimitiveState { topology: PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None, multisample: MultisampleState::default(), multiview_mask: None, cache: None,
        });

        // ── Uniforms ──
        let u = |name: &str, data: &[u8]| ctx.device.create_buffer_init(&util::BufferInitDescriptor {
            label: Some(name), contents: data, usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let u_color = u("uc", bytemuck::bytes_of(&ColorParams { ow: w, oh: h, pw, ph }));
        let u_blit = u("ub", bytemuck::bytes_of(&BlitParams { w, h }));
        let u_phase = u("up", bytemuck::bytes_of(&PhaseParams { w: pw, h: ph, amp: 0.0, al: 0.0, ah: 0.0, first: 1, _p0: 0, _p1: 0 }));

        let log2pw = (pw as f32).log2() as u32;
        let log2ph = (ph as f32).log2() as u32;
        let u_fft_rf = u("ufrf", bytemuck::bytes_of(&FftParams { n: pw, log2n: log2pw, num: ph, inv: 0, stride: 1, fft_stride: pw, _p0: 0, _p1: 0 }));
        let u_fft_cf = u("ufcf", bytemuck::bytes_of(&FftParams { n: ph, log2n: log2ph, num: pw, inv: 0, stride: pw, fft_stride: 1, _p0: 0, _p1: 0 }));
        let u_fft_ri = u("ufri", bytemuck::bytes_of(&FftParams { n: pw, log2n: log2pw, num: ph, inv: 1, stride: 1, fft_stride: pw, _p0: 0, _p1: 0 }));
        let u_fft_ci = u("ufci", bytemuck::bytes_of(&FftParams { n: ph, log2n: log2ph, num: pw, inv: 1, stride: pw, fft_stride: 1, _p0: 0, _p1: 0 }));

        let u_filt: [Buffer; N_BAND] = std::array::from_fn(|i| {
            let scale = (i / N_ORIENT as usize) as u32;
            let orient = (i % N_ORIENT as usize) as u32;
            u(&format!("uf{}", i), bytemuck::bytes_of(&FilterParams {
                w: pw, h: ph, norient: N_ORIENT, oidx: orient, scale, nscales: N_SCALES, ftype: 0, mode: 0,
            }))
        });
        let u_filt_acc: [Buffer; N_BAND] = std::array::from_fn(|i| {
            let scale = (i / N_ORIENT as usize) as u32;
            let orient = (i % N_ORIENT as usize) as u32;
            u(&format!("ua{}", i), bytemuck::bytes_of(&FilterParams {
                w: pw, h: ph, norient: N_ORIENT, oidx: orient, scale, nscales: N_SCALES, ftype: 0, mode: 0,
            }))
        });
        let u_res_hi = u("urh", bytemuck::bytes_of(&FilterParams { w: pw, h: ph, norient: N_ORIENT, oidx: 0, scale: 0, nscales: N_SCALES, ftype: 1, mode: 1 }));
        let u_res_lo = u("url", bytemuck::bytes_of(&FilterParams { w: pw, h: ph, norient: N_ORIENT, oidx: 0, scale: 0, nscales: N_SCALES, ftype: 2, mode: 1 }));

        Self {
            ow: w, oh: h, pw, ph,
            pl_r2y, pl_y2r, pl_fft, pl_filt, pl_filt_acc, pl_phase, pl_blit, surface,
            b_rgba_in, b_rgba_out, b_y, b_i, b_q, b_zeros,
            b_spec_re, b_spec_im, b_acc_re, b_acc_im, b_out_y, b_discard_im,
            b_tmp_re, b_tmp_im,
            b_ss_re, b_ss_im, b_sp_re, b_sp_im, b_sm_re, b_sm_im, b_ms_re, b_ms_im,
            prev_re, prev_im, lp_hi, lp_lo,
            u_color, u_blit, u_phase, u_filt, u_filt_acc, u_res_hi, u_res_lo,
            u_fft_rf, u_fft_cf, u_fft_ri, u_fft_ci,
            first_frame: true,
        }
    }

    pub fn process_and_render(
        &mut self, ctx: &GpuContext, frame: &[u32],
        amp: f32, freq_lo: f32, freq_hi: f32, fps: f32,
    ) {
        let (pw, ph, w, h) = (self.pw, self.ph, self.ow, self.oh);
        let two_pi = 2.0 * std::f32::consts::PI;
        let al = 1.0 - (-two_pi * freq_lo / fps).exp();
        let ah = 1.0 - (-two_pi * freq_hi / fps).exp();

        ctx.queue.write_buffer(&self.b_rgba_in, 0, bytemuck::cast_slice(frame));
        ctx.queue.write_buffer(&self.u_phase, 0, bytemuck::bytes_of(&PhaseParams {
            w: pw, h: ph, amp, al, ah, first: if self.first_frame { 1 } else { 0 }, _p0: 0, _p1: 0,
        }));

        let tex = match self.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(t) | CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => return,
        };
        let view = tex.texture.create_view(&TextureViewDescriptor::default());
        let dev = &ctx.device;

        let mut enc = dev.create_command_encoder(&CommandEncoderDescriptor { label: Some("st") });

        // 1. RGBA → Y,I,Q
        self.dispatch(dev, &mut enc, &self.pl_r2y,
            &[&self.u_color, &self.b_rgba_in, &self.b_y, &self.b_i, &self.b_q],
            (pw.div_ceil(16), ph.div_ceil(16), 1));

        // 2. Forward 2D FFT of Y
        self.fft_pass(dev, &mut enc, &self.u_fft_rf, &self.b_y, &self.b_zeros, &self.b_tmp_re, &self.b_tmp_im, ph);
        self.fft_pass(dev, &mut enc, &self.u_fft_cf, &self.b_tmp_re, &self.b_tmp_im, &self.b_spec_re, &self.b_spec_im, pw);

        // 3. Clear accumulator
        enc.copy_buffer_to_buffer(&self.b_zeros, 0, &self.b_acc_re, 0, (pw*ph*4) as u64);
        enc.copy_buffer_to_buffer(&self.b_zeros, 0, &self.b_acc_im, 0, (pw*ph*4) as u64);

        // 4. Process each bandpass sub-band
        let wg = (pw.div_ceil(16), ph.div_ceil(16), 1);
        for idx in 0..N_BAND {
            // 4a. Apply analysis filter: spec → sub_spec
            self.dispatch(dev, &mut enc, &self.pl_filt,
                &[&self.u_filt[idx], &self.b_spec_re, &self.b_spec_im, &self.b_ss_re, &self.b_ss_im], wg);

            // 4b. Inverse 2D FFT: sub_spec → sub_spatial
            self.fft_pass(dev, &mut enc, &self.u_fft_ci, &self.b_ss_re, &self.b_ss_im, &self.b_tmp_re, &self.b_tmp_im, pw);
            self.fft_pass(dev, &mut enc, &self.u_fft_ri, &self.b_tmp_re, &self.b_tmp_im, &self.b_sp_re, &self.b_sp_im, ph);

            // 4c. Phase amplification
            self.dispatch(dev, &mut enc, &self.pl_phase,
                &[&self.u_phase, &self.b_sp_re, &self.b_sp_im,
                  &self.prev_re[idx], &self.prev_im[idx], &self.lp_hi[idx], &self.lp_lo[idx],
                  &self.b_sm_re, &self.b_sm_im], wg);

            // 4d. Forward 2D FFT: modified → mod_spec
            self.fft_pass(dev, &mut enc, &self.u_fft_rf, &self.b_sm_re, &self.b_sm_im, &self.b_tmp_re, &self.b_tmp_im, ph);
            self.fft_pass(dev, &mut enc, &self.u_fft_cf, &self.b_tmp_re, &self.b_tmp_im, &self.b_ms_re, &self.b_ms_im, pw);

            // 4e. Filter × accumulate
            self.dispatch(dev, &mut enc, &self.pl_filt_acc,
                &[&self.u_filt_acc[idx], &self.b_ms_re, &self.b_ms_im, &self.b_acc_re, &self.b_acc_im], wg);
        }

        // 5. Residuals (H² accumulate, no phase)
        self.dispatch(dev, &mut enc, &self.pl_filt_acc,
            &[&self.u_res_hi, &self.b_spec_re, &self.b_spec_im, &self.b_acc_re, &self.b_acc_im], wg);
        self.dispatch(dev, &mut enc, &self.pl_filt_acc,
            &[&self.u_res_lo, &self.b_spec_re, &self.b_spec_im, &self.b_acc_re, &self.b_acc_im], wg);

        // 6. Inverse 2D FFT: accum → output Y
        self.fft_pass(dev, &mut enc, &self.u_fft_ci, &self.b_acc_re, &self.b_acc_im, &self.b_tmp_re, &self.b_tmp_im, pw);
        self.fft_pass(dev, &mut enc, &self.u_fft_ri, &self.b_tmp_re, &self.b_tmp_im, &self.b_out_y, &self.b_discard_im, ph);

        // 7. YIQ → RGBA
        self.dispatch(dev, &mut enc, &self.pl_y2r,
            &[&self.u_color, &self.b_out_y, &self.b_i, &self.b_q, &self.b_rgba_out],
            (w.div_ceil(16), h.div_ceil(16), 1));

        // 8. Blit
        {
            let bg = dev.create_bind_group(&BindGroupDescriptor {
                label: Some("blit"), layout: &self.pl_blit.get_bind_group_layout(0),
                entries: &[
                    BindGroupEntry { binding: 0, resource: self.u_blit.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: self.b_rgba_out.as_entire_binding() },
                ],
            });
            let mut pass = enc.begin_render_pass(&RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &view, resolve_target: None,
                    ops: Operations { load: LoadOp::Clear(Color::BLACK), store: StoreOp::Store },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None, timestamp_writes: None,
                occlusion_query_set: None, multiview_mask: None,
            });
            pass.set_pipeline(&self.pl_blit);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.draw(0..6, 0..1);
        }

        ctx.queue.submit(std::iter::once(enc.finish()));
        tex.present();
        self.first_frame = false;
    }

    fn dispatch(&self, dev: &Device, enc: &mut CommandEncoder, pipeline: &ComputePipeline,
                bufs: &[&Buffer], wg: (u32, u32, u32)) {
        let entries: Vec<BindGroupEntry> = bufs.iter().enumerate()
            .map(|(i, b)| BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        let bg = dev.create_bind_group(&BindGroupDescriptor {
            label: None, layout: &pipeline.get_bind_group_layout(0), entries: &entries,
        });
        let mut pass = enc.begin_compute_pass(&ComputePassDescriptor { label: None, timestamp_writes: None });
        GpuContext::record_dispatch(&mut pass, pipeline, &bg, wg);
    }

    fn fft_pass(&self, dev: &Device, enc: &mut CommandEncoder,
                params: &Buffer, in_re: &Buffer, in_im: &Buffer,
                out_re: &Buffer, out_im: &Buffer, num_ffts: u32) {
        self.dispatch(dev, enc, &self.pl_fft,
            &[params, in_re, in_im, out_re, out_im], (num_ffts, 1, 1));
    }
}

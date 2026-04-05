// steerable.rs — Complex Steerable Pyramid phase-based motion magnification
//
// Per frame (~36 GPU dispatches at 2s×2o):
// 1. RGBA → Y,I,Q (pad)         2. FFT(Y)
// 3. For each bandpass sub-band: apply_filter → IFFT → phase_amplify → FFT → filter_accum
// 4. Residuals: filter_accum     5. IFFT(accum) → modified Y
// 6. YIQ → RGBA                  7. Blit to canvas
//
// Optimized: pre-created bind groups, pre-computed filter textures,
// 2 compute passes per frame.

use crate::gpu::GpuContext;
use bytemuck::{Pod, Zeroable};
use wasm_bindgen::JsCast;
use wgpu::util::DeviceExt;
use wgpu::*;

fn next_pow2(n: u32) -> u32 {
    if n.is_power_of_two() { n } else { n.next_power_of_two() }
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
    pub ow: u32, pub oh: u32,
    pub pw: u32, pub ph: u32,
    pub n_scales: u32, pub n_orient: u32, pub n_band: usize,

    // Compute pipelines
    pl_r2y: ComputePipeline,
    pl_y2r: ComputePipeline,
    pl_fft: ComputePipeline,
    pl_apply: ComputePipeline,
    pl_accum: ComputePipeline,
    pl_phase: ComputePipeline,
    pl_blit: RenderPipeline,
    surface: Surface<'static>,

    // Buffers that need CPU writes each frame
    b_rgba_in: Buffer,
    u_phase: Buffer,
    // Buffers for accumulator clear (encoder-level copy)
    b_zeros: Buffer,
    b_acc_re: Buffer,
    b_acc_im: Buffer,

    // Pre-created bind groups
    bg_r2y: BindGroup,
    bg_fft_y_row: BindGroup,
    bg_fft_y_col: BindGroup,
    bg_apply: Vec<BindGroup>,
    bg_ifft_sb_col: BindGroup,
    bg_ifft_sb_row: BindGroup,
    bg_phase: Vec<BindGroup>,
    bg_fft_mod_row: BindGroup,
    bg_fft_mod_col: BindGroup,
    bg_accum: Vec<BindGroup>,
    bg_res: BindGroup,
    bg_ifft_final_col: BindGroup,
    bg_ifft_final_row: BindGroup,
    bg_y2r: BindGroup,
    bg_blit: BindGroup,

    first_frame: bool,
}

impl SteerablePipeline {
    pub fn new(ctx: &GpuContext, w: u32, h: u32, max_fft: u32, n_scales: u32, n_orient: u32) -> Self {
        assert!(n_scales >= 1 && n_scales <= 4);
        assert!(n_orient >= 1 && n_orient <= 8);
        let n_band = (n_scales * n_orient) as usize;
        let pw = next_pow2(w).min(max_fft);
        let ph = next_pow2(h).min(max_fft);
        let ppix = (pw * ph) as u64;
        let opix = (w * h) as u64;
        log::info!("Steerable: {}x{} pad {}x{}", w, h, pw, ph);

        let s = BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST;
        let mk = |name: &str, sz: u64| ctx.create_buffer(name, sz, s);
        let dev = &ctx.device;

        // ── Buffers ──
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

        let prev_re: Vec<Buffer> = (0..n_band).map(|i| mk(&format!("pr{}", i), ppix*4)).collect();
        let prev_im: Vec<Buffer> = (0..n_band).map(|i| mk(&format!("pi{}", i), ppix*4)).collect();
        let lp_hi: Vec<Buffer> = (0..n_band).map(|i| mk(&format!("lh{}", i), ppix*4)).collect();
        let lp_lo: Vec<Buffer> = (0..n_band).map(|i| mk(&format!("ll{}", i), ppix*4)).collect();

        // Pre-computed filter buffers (computed once at init)
        let filt_band: Vec<Buffer> = (0..n_band).map(|i| mk(&format!("fb{}", i), ppix*4)).collect();
        let filt_hi = mk("fhi", ppix * 4);
        let filt_lo = mk("flo", ppix * 4);
        let filt_res = mk("frs", ppix * 4); // filt_hi + filt_lo (summed at init)

        // ── Compute pipelines ──
        let mkc = |label: &str, src: &str| {
            let m = dev.create_shader_module(ShaderModuleDescriptor {
                label: Some(label), source: ShaderSource::Wgsl(src.into()),
            });
            dev.create_compute_pipeline(&ComputePipelineDescriptor {
                label: Some(label), layout: None, module: &m, entry_point: Some("main"),
                compilation_options: Default::default(), cache: None,
            })
        };

        let pl_r2y = mkc("r2y", include_str!("shaders/rgba_to_y.wgsl"));
        let pl_y2r = mkc("y2r", include_str!("shaders/yiq_to_rgba.wgsl"));
        let fft_src = include_str!("shaders/fft.wgsl").replace("/*SHARED_SIZE*/", &max_fft.to_string());
        let pl_fft = mkc("fft", &fft_src);
        let pl_precomp = mkc("precomp", include_str!("shaders/precompute_filter.wgsl"));
        let pl_apply = mkc("apply", include_str!("shaders/apply_filter.wgsl"));
        let pl_accum = mkc("accum", include_str!("shaders/apply_filter_accum.wgsl"));
        let pl_phase = mkc("phase", include_str!("shaders/phase_amplify.wgsl"));
        let pl_add = mkc("add", include_str!("shaders/add_buffers.wgsl"));

        // ── Surface + blit ──
        let canvas: web_sys::HtmlCanvasElement = web_sys::window().unwrap()
            .document().unwrap().get_element_by_id("output").unwrap().unchecked_into();
        canvas.set_width(w); canvas.set_height(h);
        let surface = ctx.instance.create_surface(wgpu::SurfaceTarget::Canvas(canvas)).unwrap();
        let caps = surface.get_capabilities(&ctx.adapter);
        let fmt = caps.formats[0];
        surface.configure(dev, &SurfaceConfiguration {
            usage: TextureUsages::RENDER_ATTACHMENT, format: fmt,
            width: w, height: h,
            present_mode: caps.present_modes[0],
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        });

        let blit_mod = dev.create_shader_module(ShaderModuleDescriptor {
            label: Some("blit"), source: ShaderSource::Wgsl(include_str!("shaders/blit.wgsl").into()),
        });
        let pl_blit = dev.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some("blit"), layout: None,
            vertex: VertexState { module: &blit_mod, entry_point: Some("vs_main"), buffers: &[], compilation_options: Default::default() },
            fragment: Some(FragmentState { module: &blit_mod, entry_point: Some("fs_main"),
                targets: &[Some(ColorTargetState { format: fmt, blend: None, write_mask: ColorWrites::ALL })],
                compilation_options: Default::default() }),
            primitive: PrimitiveState { topology: PrimitiveTopology::TriangleList, ..Default::default() },
            depth_stencil: None, multisample: MultisampleState::default(), multiview_mask: None, cache: None,
        });

        // ── Uniforms ──
        let u = |name: &str, data: &[u8]| dev.create_buffer_init(&util::BufferInitDescriptor {
            label: Some(name), contents: data, usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
        });

        let u_color = u("uc", bytemuck::bytes_of(&ColorParams { ow: w, oh: h, pw, ph }));
        let u_blit = u("ub", bytemuck::bytes_of(&BlitParams { w, h }));
        let u_phase = u("up", bytemuck::bytes_of(&PhaseParams { w: pw, h: ph, amp: 0.0, al: 0.0, ah: 0.0, first: 1, _p0: 0, _p1: 0 }));
        let u_pdims = u("ud", bytemuck::bytes_of(&BlitParams { w: pw, h: ph }));

        let log2pw = (pw as f32).log2() as u32;
        let log2ph = (ph as f32).log2() as u32;
        let u_fft_rf = u("ufrf", bytemuck::bytes_of(&FftParams { n: pw, log2n: log2pw, num: ph, inv: 0, stride: 1, fft_stride: pw, _p0: 0, _p1: 0 }));
        let u_fft_cf = u("ufcf", bytemuck::bytes_of(&FftParams { n: ph, log2n: log2ph, num: pw, inv: 0, stride: pw, fft_stride: 1, _p0: 0, _p1: 0 }));
        let u_fft_ri = u("ufri", bytemuck::bytes_of(&FftParams { n: pw, log2n: log2pw, num: ph, inv: 1, stride: 1, fft_stride: pw, _p0: 0, _p1: 0 }));
        let u_fft_ci = u("ufci", bytemuck::bytes_of(&FftParams { n: ph, log2n: log2ph, num: pw, inv: 1, stride: pw, fft_stride: 1, _p0: 0, _p1: 0 }));

        // ── Pre-compute filter textures ──
        // Build filter param uniforms (temporary — only needed for precomputation)
        let u_filt_params: Vec<Buffer> = (0..n_band).map(|i| {
            let scale = (i / n_orient as usize) as u32;
            let orient = (i % n_orient as usize) as u32;
            u(&format!("ufp{}", i), bytemuck::bytes_of(&FilterParams {
                w: pw, h: ph, norient: n_orient, oidx: orient, scale, nscales: n_scales, ftype: 0, mode: 0,
            }))
        }).collect();
        let u_res_hi_p = u("urh", bytemuck::bytes_of(&FilterParams {
            w: pw, h: ph, norient: n_orient, oidx: 0, scale: 0, nscales: n_scales, ftype: 1, mode: 1,
        }));
        let u_res_lo_p = u("url", bytemuck::bytes_of(&FilterParams {
            w: pw, h: ph, norient: n_orient, oidx: 0, scale: 0, nscales: n_scales, ftype: 2, mode: 1,
        }));

        // Create precompute bind groups (temporary)
        let mk_precomp_bg = |params_buf: &Buffer, filter_buf: &Buffer| -> BindGroup {
            dev.create_bind_group(&BindGroupDescriptor {
                label: None, layout: &pl_precomp.get_bind_group_layout(0),
                entries: &[
                    BindGroupEntry { binding: 0, resource: params_buf.as_entire_binding() },
                    BindGroupEntry { binding: 1, resource: filter_buf.as_entire_binding() },
                ],
            })
        };
        let precomp_bgs: Vec<BindGroup> = (0..n_band).map(|i| {
            mk_precomp_bg(&u_filt_params[i], &filt_band[i])
        }).collect();
        let bg_precomp_hi = mk_precomp_bg(&u_res_hi_p, &filt_hi);
        let bg_precomp_lo = mk_precomp_bg(&u_res_lo_p, &filt_lo);

        // Dispatch precomputation
        let wg = (pw.div_ceil(16), ph.div_ceil(16), 1);
        {
            let mut enc = dev.create_command_encoder(&CommandEncoderDescriptor { label: Some("precomp") });
            {
                let bg_add = {
                    let entries: Vec<BindGroupEntry> = [&u_pdims as &Buffer, &filt_hi, &filt_lo, &filt_res]
                        .iter().enumerate()
                        .map(|(i, b)| BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
                        .collect();
                    dev.create_bind_group(&BindGroupDescriptor {
                        label: None, layout: &pl_add.get_bind_group_layout(0), entries: &entries,
                    })
                };
                let mut p = enc.begin_compute_pass(&ComputePassDescriptor { label: None, timestamp_writes: None });
                for bg in &precomp_bgs {
                    p.set_pipeline(&pl_precomp);
                    p.set_bind_group(0, Some(bg), &[]);
                    p.dispatch_workgroups(wg.0, wg.1, wg.2);
                }
                p.set_pipeline(&pl_precomp);
                p.set_bind_group(0, Some(&bg_precomp_hi), &[]);
                p.dispatch_workgroups(wg.0, wg.1, wg.2);
                p.set_pipeline(&pl_precomp);
                p.set_bind_group(0, Some(&bg_precomp_lo), &[]);
                p.dispatch_workgroups(wg.0, wg.1, wg.2);
                // Sum filt_hi + filt_lo → filt_res
                p.set_pipeline(&pl_add);
                p.set_bind_group(0, Some(&bg_add), &[]);
                p.dispatch_workgroups(wg.0, wg.1, wg.2);
            }
            ctx.queue.submit(std::iter::once(enc.finish()));
        }
        log::info!("Pre-computed {} filter textures", n_band + 2);

        // ── Pre-create per-frame bind groups ──
        let mk_bg = |pipeline: &ComputePipeline, bufs: &[&Buffer]| -> BindGroup {
            let entries: Vec<BindGroupEntry> = bufs.iter().enumerate()
                .map(|(i, b)| BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
                .collect();
            dev.create_bind_group(&BindGroupDescriptor {
                label: None, layout: &pipeline.get_bind_group_layout(0), entries: &entries,
            })
        };

        // Pass 1: color convert + forward FFT
        let bg_r2y = mk_bg(&pl_r2y, &[&u_color, &b_rgba_in, &b_y, &b_i, &b_q]);
        let bg_fft_y_row = mk_bg(&pl_fft, &[&u_fft_rf, &b_y, &b_zeros, &b_tmp_re, &b_tmp_im]);
        let bg_fft_y_col = mk_bg(&pl_fft, &[&u_fft_cf, &b_tmp_re, &b_tmp_im, &b_spec_re, &b_spec_im]);

        // Per sub-band: apply filter (analysis)
        let bg_apply: Vec<BindGroup> = (0..n_band).map(|i| {
            mk_bg(&pl_apply, &[&u_pdims, &filt_band[i], &b_spec_re, &b_spec_im, &b_ss_re, &b_ss_im])
        }).collect();
        let bg_ifft_sb_col = mk_bg(&pl_fft, &[&u_fft_ci, &b_ss_re, &b_ss_im, &b_tmp_re, &b_tmp_im]);
        let bg_ifft_sb_row = mk_bg(&pl_fft, &[&u_fft_ri, &b_tmp_re, &b_tmp_im, &b_sp_re, &b_sp_im]);
        let bg_phase: Vec<BindGroup> = (0..n_band).map(|i| {
            mk_bg(&pl_phase, &[&u_phase, &b_sp_re, &b_sp_im,
                &prev_re[i], &prev_im[i], &lp_hi[i], &lp_lo[i],
                &b_sm_re, &b_sm_im])
        }).collect();
        let bg_fft_mod_row = mk_bg(&pl_fft, &[&u_fft_rf, &b_sm_re, &b_sm_im, &b_tmp_re, &b_tmp_im]);
        let bg_fft_mod_col = mk_bg(&pl_fft, &[&u_fft_cf, &b_tmp_re, &b_tmp_im, &b_ms_re, &b_ms_im]);

        // Per sub-band: filter accumulate (reconstruction)
        let bg_accum: Vec<BindGroup> = (0..n_band).map(|i| {
            mk_bg(&pl_accum, &[&u_pdims, &filt_band[i], &b_ms_re, &b_ms_im, &b_acc_re, &b_acc_im])
        }).collect();

        // Residual (pre-summed filt_hi² + filt_lo² into single buffer)
        let bg_res = mk_bg(&pl_accum, &[&u_pdims, &filt_res, &b_spec_re, &b_spec_im, &b_acc_re, &b_acc_im]);

        // Final reconstruction
        let bg_ifft_final_col = mk_bg(&pl_fft, &[&u_fft_ci, &b_acc_re, &b_acc_im, &b_tmp_re, &b_tmp_im]);
        let bg_ifft_final_row = mk_bg(&pl_fft, &[&u_fft_ri, &b_tmp_re, &b_tmp_im, &b_out_y, &b_discard_im]);
        let bg_y2r = mk_bg(&pl_y2r, &[&u_color, &b_out_y, &b_i, &b_q, &b_rgba_out]);

        // Blit (render pipeline)
        let bg_blit = dev.create_bind_group(&BindGroupDescriptor {
            label: Some("blit"), layout: &pl_blit.get_bind_group_layout(0),
            entries: &[
                BindGroupEntry { binding: 0, resource: u_blit.as_entire_binding() },
                BindGroupEntry { binding: 1, resource: b_rgba_out.as_entire_binding() },
            ],
        });

        Self {
            ow: w, oh: h, pw, ph, n_scales, n_orient, n_band,
            pl_r2y, pl_y2r, pl_fft, pl_apply, pl_accum, pl_phase, pl_blit, surface,
            b_rgba_in, u_phase, b_zeros, b_acc_re, b_acc_im,
            bg_r2y, bg_fft_y_row, bg_fft_y_col,
            bg_apply, bg_ifft_sb_col, bg_ifft_sb_row,
            bg_phase, bg_fft_mod_row, bg_fft_mod_col,
            bg_accum, bg_res,
            bg_ifft_final_col, bg_ifft_final_row, bg_y2r, bg_blit,
            first_frame: true,
        }
    }

    /// Reset temporal state (call on camera switch even if resolution unchanged).
    pub fn reset_state(&mut self) {
        self.first_frame = true;
    }

    pub fn process_and_render(
        &mut self, ctx: &GpuContext, frame: &[u8],
        amp: f32, freq_lo: f32, freq_hi: f32, fps: f32,
    ) {
        let (pw, ph, w, h) = (self.pw, self.ph, self.ow, self.oh);
        let two_pi = 2.0 * std::f32::consts::PI;
        let al = 1.0 - (-two_pi * freq_lo / fps).exp();
        let ah = 1.0 - (-two_pi * freq_hi / fps).exp();

        ctx.queue.write_buffer(&self.b_rgba_in, 0, frame);
        ctx.queue.write_buffer(&self.u_phase, 0, bytemuck::bytes_of(&PhaseParams {
            w: pw, h: ph, amp, al, ah, first: if self.first_frame { 1 } else { 0 }, _p0: 0, _p1: 0,
        }));

        let tex = match self.surface.get_current_texture() {
            CurrentSurfaceTexture::Success(t) | CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => return,
        };
        let view = tex.texture.create_view(&TextureViewDescriptor::default());
        let mut enc = ctx.device.create_command_encoder(&CommandEncoderDescriptor { label: Some("st") });

        let wg = (pw.div_ceil(16), ph.div_ceil(16), 1);

        // ── Compute pass 1: color convert + forward FFT ──
        {
            let mut p = enc.begin_compute_pass(&ComputePassDescriptor { label: None, timestamp_writes: None });
            Self::rec(&mut p, &self.pl_r2y, &self.bg_r2y, (pw.div_ceil(16), ph.div_ceil(16), 1));
            Self::rec(&mut p, &self.pl_fft, &self.bg_fft_y_row, (ph, 1, 1));
            Self::rec(&mut p, &self.pl_fft, &self.bg_fft_y_col, (pw, 1, 1));
        }

        // Clear accumulators (encoder-level copy, between compute passes)
        let copy_sz = (pw * ph * 4) as u64;
        enc.copy_buffer_to_buffer(&self.b_zeros, 0, &self.b_acc_re, 0, copy_sz);
        enc.copy_buffer_to_buffer(&self.b_zeros, 0, &self.b_acc_im, 0, copy_sz);

        // ── Compute pass 2: sub-band processing + reconstruction ──
        {
            let mut p = enc.begin_compute_pass(&ComputePassDescriptor { label: None, timestamp_writes: None });

            for idx in 0..self.n_band {
                // Analysis: apply pre-computed filter
                Self::rec(&mut p, &self.pl_apply, &self.bg_apply[idx], wg);
                // Inverse 2D FFT: sub-band spectrum → spatial
                Self::rec(&mut p, &self.pl_fft, &self.bg_ifft_sb_col, (pw, 1, 1));
                Self::rec(&mut p, &self.pl_fft, &self.bg_ifft_sb_row, (ph, 1, 1));
                // Phase amplification
                Self::rec(&mut p, &self.pl_phase, &self.bg_phase[idx], wg);
                // Forward 2D FFT: modified spatial → spectrum
                Self::rec(&mut p, &self.pl_fft, &self.bg_fft_mod_row, (ph, 1, 1));
                Self::rec(&mut p, &self.pl_fft, &self.bg_fft_mod_col, (pw, 1, 1));
                // Reconstruction: apply pre-computed filter and accumulate
                Self::rec(&mut p, &self.pl_accum, &self.bg_accum[idx], wg);
            }

            // Residual (pre-summed hi² + lo² in single buffer)
            Self::rec(&mut p, &self.pl_accum, &self.bg_res, wg);

            // Inverse 2D FFT: accumulated spectrum → modified Y
            Self::rec(&mut p, &self.pl_fft, &self.bg_ifft_final_col, (pw, 1, 1));
            Self::rec(&mut p, &self.pl_fft, &self.bg_ifft_final_row, (ph, 1, 1));

            // YIQ → RGBA
            Self::rec(&mut p, &self.pl_y2r, &self.bg_y2r, (w.div_ceil(16), h.div_ceil(16), 1));
        }

        // ── Render pass: blit to canvas ──
        {
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
            pass.set_bind_group(0, Some(&self.bg_blit), &[]);
            pass.draw(0..6, 0..1);
        }

        ctx.queue.submit(std::iter::once(enc.finish()));
        tex.present();
        self.first_frame = false;
    }

    fn rec<'a>(pass: &mut ComputePass<'a>, pl: &'a ComputePipeline, bg: &'a BindGroup, wg: (u32, u32, u32)) {
        pass.set_pipeline(pl);
        pass.set_bind_group(0, Some(bg), &[]);
        pass.dispatch_workgroups(wg.0, wg.1, wg.2);
    }
}

// lib.rs — Minimal test: just render solid color to WebGPU canvas

mod gpu;
mod evm;
mod video;

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use gpu::GpuContext;

fn perf_now() -> f64 {
    web_sys::window().unwrap().performance().unwrap().now()
}

async fn next_frame() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        web_sys::window().unwrap().request_animation_frame(&resolve).unwrap();
    });
    wasm_bindgen_futures::JsFuture::from(promise).await.unwrap();
}

#[wasm_bindgen(start)]
pub async fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).unwrap();

    log::info!("Minimal surface test...");

    let ctx = GpuContext::new().await;
    log::info!("Device ready");

    // Configure surface at canvas's current size
    let canvas: web_sys::HtmlCanvasElement = web_sys::window().unwrap()
        .document().unwrap()
        .get_element_by_id("output").unwrap()
        .dyn_into().unwrap();
    let w = canvas.width();
    let h = canvas.height();
    log::info!("Canvas: {}x{}", w, h);

    let caps = ctx.surface.get_capabilities(&ctx.adapter);
    log::info!("Caps: {:?}", caps.formats);
    let format = caps.formats[0];

    ctx.surface.configure(&ctx.device, &wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: w,
        height: h,
        present_mode: caps.present_modes[0],
        desired_maximum_frame_latency: 2,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
    });
    log::info!("Surface configured");

    // Minimal render pipeline — just clears to magenta
    wasm_bindgen_futures::spawn_local(async move {
        loop {
            next_frame().await;

            let frame = ctx.surface.get_current_texture();
            let tex = match frame {
                wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
                _ => { log::error!("No surface texture"); continue; }
            };
            let view = tex.texture.create_view(&wgpu::TextureViewDescriptor::default());

            let mut encoder = ctx.device.create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("clear") }
            );
            {
                let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("clear"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color { r: 1.0, g: 0.0, b: 1.0, a: 1.0 }),
                            store: wgpu::StoreOp::Store,
                        },
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                // No draw call — just clear
            }
            ctx.queue.submit(std::iter::once(encoder.finish()));
            tex.present();
        }
    });

    Ok(())
}

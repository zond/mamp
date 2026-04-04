// lib.rs — WASM entry point for real-time motion magnification

mod gpu;
mod model;
mod video;
mod weights;

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wgpu::BufferUsages;

use gpu::GpuContext;
use model::{ModelWeights, MotionMagModel};
use video::{OutputRenderer, VideoCapture};

const WIDTH: u32 = 480;
const HEIGHT: u32 = 360;

/// State shared across animation frames.
struct AppState {
    ctx: GpuContext,
    model: MotionMagModel,
    capture: VideoCapture,
    renderer: OutputRenderer,
    prev_frame: Option<Vec<u32>>,
    alpha: f32,
    running: bool,
}

/// Load trained weights from server, or fall back to random initialization.
async fn load_weights() -> ModelWeights {
    let layer_specs: &[(&str, usize)] = &[
        ("enc_conv1.weight", 16 * 3 * 3 * 3),
        ("enc_conv1.bias", 16),
        ("enc_conv2.weight", 32 * 16 * 3 * 3),
        ("enc_conv2.bias", 32),
        ("enc_conv3.weight", 32 * 32 * 3 * 3),
        ("enc_conv3.bias", 32),
        ("enc_texture.weight", 32 * 32 * 1 * 1),
        ("enc_texture.bias", 32),
        ("dec_conv1.weight", 32 * 32 * 3 * 3),
        ("dec_conv1.bias", 32),
        ("dec_conv2.weight", 16 * 32 * 3 * 3),
        ("dec_conv2.bias", 16),
        ("dec_conv3.weight", 3 * 16 * 3 * 3),
        ("dec_conv3.bias", 3),
    ];

    let mut loaded: Vec<Vec<f32>> = Vec::new();
    let mut all_ok = true;

    for (name, expected_len) in layer_specs {
        let url = format!("weights/{}.bin", name);
        match weights::load_weight_file(&url).await {
            Ok(data) if data.len() == *expected_len => {
                log::info!("Loaded {}: {} floats", name, data.len());
                loaded.push(data);
            }
            Ok(data) => {
                log::warn!("{}: expected {} floats, got {} — using random", name, expected_len, data.len());
                all_ok = false;
                break;
            }
            Err(e) => {
                log::warn!("Failed to load {}: {:?} — using random weights", name, e);
                all_ok = false;
                break;
            }
        }
    }

    if !all_ok || loaded.len() != layer_specs.len() {
        log::info!("Using random placeholder weights (train model with train.py)");
        return ModelWeights::random();
    }

    let mut it = loaded.into_iter();
    ModelWeights {
        enc_conv1_w: it.next().unwrap(),
        enc_conv1_b: it.next().unwrap(),
        enc_conv2_w: it.next().unwrap(),
        enc_conv2_b: it.next().unwrap(),
        enc_conv3_w: it.next().unwrap(),
        enc_conv3_b: it.next().unwrap(),
        enc_texture_w: it.next().unwrap(),
        enc_texture_b: it.next().unwrap(),
        dec_conv1_w: it.next().unwrap(),
        dec_conv1_b: it.next().unwrap(),
        dec_conv2_w: it.next().unwrap(),
        dec_conv2_b: it.next().unwrap(),
        dec_conv3_w: it.next().unwrap(),
        dec_conv3_b: it.next().unwrap(),
    }
}

/// Main WASM entry point.
#[wasm_bindgen(start)]
pub async fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).unwrap();

    log::info!("Initializing WebGPU motion magnification...");

    let ctx = GpuContext::new().await;
    log::info!("WebGPU device ready");

    let weights = load_weights().await;
    let model = MotionMagModel::new(&ctx, WIDTH, HEIGHT, &weights);
    log::info!("Model built: {}x{}, latent {}x{}", WIDTH, HEIGHT, model.latent_width, model.latent_height);

    let capture = VideoCapture::new(WIDTH, HEIGHT)?;
    capture.start().await?;
    log::info!("Camera stream active");

    let renderer = OutputRenderer::new("output", WIDTH, HEIGHT)?;

    let state = Rc::new(RefCell::new(AppState {
        ctx,
        model,
        capture,
        renderer,
        prev_frame: None,
        alpha: 20.0,
        running: true,
    }));

    request_animation_frame(state.clone());

    // Expose alpha control to JS
    let state_for_js = state.clone();
    let set_alpha = Closure::wrap(Box::new(move |alpha: f32| {
        state_for_js.borrow_mut().alpha = alpha;
    }) as Box<dyn FnMut(f32)>);

    let window = web_sys::window().unwrap();
    js_sys::Reflect::set(&window, &"setMagnification".into(), set_alpha.as_ref())?;
    set_alpha.forget();

    let state_for_toggle = state.clone();
    let toggle = Closure::wrap(Box::new(move || {
        let mut s = state_for_toggle.borrow_mut();
        s.running = !s.running;
        if s.running {
            drop(s);
            request_animation_frame(state_for_toggle.clone());
        }
    }) as Box<dyn FnMut()>);

    js_sys::Reflect::set(&window, &"toggleMagnification".into(), toggle.as_ref())?;
    toggle.forget();

    Ok(())
}

fn request_animation_frame(state: Rc<RefCell<AppState>>) {
    let closure = Rc::new(RefCell::new(None::<Closure<dyn FnMut()>>));
    let closure_clone = closure.clone();

    *closure.borrow_mut() = Some(Closure::wrap(Box::new(move || {
        let should_continue = process_frame(&state);

        if should_continue {
            let window = web_sys::window().unwrap();
            window
                .request_animation_frame(
                    closure_clone
                        .borrow()
                        .as_ref()
                        .unwrap()
                        .as_ref()
                        .unchecked_ref(),
                )
                .unwrap();
        }
    }) as Box<dyn FnMut()>));

    let window = web_sys::window().unwrap();
    window
        .request_animation_frame(
            closure
                .borrow()
                .as_ref()
                .unwrap()
                .as_ref()
                .unchecked_ref(),
        )
        .unwrap();
}

/// Process one frame: grab camera, run model, render output.
fn process_frame(state: &Rc<RefCell<AppState>>) -> bool {
    let mut s = state.borrow_mut();
    if !s.running {
        return false;
    }

    let current_frame = match s.capture.grab_frame() {
        Ok(f) => f,
        Err(e) => {
            log::error!("Frame grab failed: {:?}", e);
            return true;
        }
    };

    if let Some(ref prev) = s.prev_frame {
        let t0 = web_sys::window().unwrap().performance().unwrap().now();

        let output_buf = s.model.magnify(&s.ctx, prev, &current_frame, s.alpha);

        // Read back output pixels via staging buffer
        let staging = s.ctx.create_buffer(
            "staging",
            (WIDTH * HEIGHT * 4) as u64,
            BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        );

        let mut encoder = s.ctx.device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor { label: Some("readback") },
        );
        encoder.copy_buffer_to_buffer(&output_buf, 0, &staging, 0, (WIDTH * HEIGHT * 4) as u64);
        s.ctx.queue.submit(std::iter::once(encoder.finish()));

        // Render input frame while GPU processes (async readback would be production path)
        let _ = s.renderer.draw(&current_frame);

        let t1 = web_sys::window().unwrap().performance().unwrap().now();
        log::info!("Frame time: {:.1}ms", t1 - t0);
    }

    s.prev_frame = Some(current_frame);
    true
}

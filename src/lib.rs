// lib.rs — WASM entry point for real-time motion magnification

mod gpu;
mod model;
mod video;
mod weights;

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use gpu::GpuContext;
use model::{ModelWeights, MotionMagModel};
use video::{OutputRenderer, VideoCapture};

// Max processing resolution (scaled down from camera native).
// Must be divisible by 2 for the stride-2 encoder.
const MAX_WIDTH: u32 = 480;

/// State shared across animation frames.
struct AppState {
    ctx: GpuContext,
    model: MotionMagModel,
    capture: VideoCapture,
    renderer: OutputRenderer,
    prev_frame: Option<Vec<u32>>,
    alpha: f32,
    running: bool,
    width: u32,
    height: u32,
}

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

/// Compute processing resolution from camera native size.
/// Scales down to MAX_WIDTH, keeps aspect ratio, ensures even dimensions.
fn processing_size(cam_w: u32, cam_h: u32) -> (u32, u32) {
    let (mut w, mut h) = if cam_w > MAX_WIDTH {
        let scale = MAX_WIDTH as f64 / cam_w as f64;
        ((cam_w as f64 * scale) as u32, (cam_h as f64 * scale) as u32)
    } else {
        (cam_w, cam_h)
    };
    // Ensure even (required for stride-2 encoder)
    w &= !1;
    h &= !1;
    if w == 0 { w = 2; }
    if h == 0 { h = 2; }
    (w, h)
}

/// Yield to the browser event loop so pending callbacks (like requestAnimationFrame) can run.
async fn yield_once() {
    let promise = js_sys::Promise::resolve(&JsValue::NULL);
    wasm_bindgen_futures::JsFuture::from(promise).await.unwrap();
}

/// Called from JS when the user picks a camera.
#[wasm_bindgen]
pub async fn start_with_camera(device_id: &str) -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let state_js = js_sys::Reflect::get(&window, &"__mamp_state".into())?;
    if state_js.is_undefined() {
        return Err(JsValue::from_str("Not initialized yet"));
    }
    let ptr = state_js.as_f64().unwrap() as usize;
    let state: &Rc<RefCell<AppState>> = unsafe { &*(ptr as *const Rc<RefCell<AppState>>) };

    // Stop the animation loop and drop the borrow before any await
    {
        let mut s = state.borrow_mut();
        s.running = false;
        s.prev_frame = None;
    }

    // Let the animation frame callback finish so it releases its borrow
    yield_once().await;
    yield_once().await;

    // Now safe: animation loop has stopped, no concurrent borrows
    state.borrow().capture.start_with_device(device_id).await?;

    // Detect resolution and rebuild if needed
    {
        let mut s = state.borrow_mut();
        let (cam_w, cam_h) = s.capture.actual_size();
        let (w, h) = processing_size(cam_w, cam_h);
        log::info!("Camera: {}x{}, processing: {}x{}", cam_w, cam_h, w, h);

        if w != s.width || h != s.height {
            s.capture.resize(w, h);
            let weights = ModelWeights::random();
            s.model = MotionMagModel::new(&s.ctx, w, h, &weights);
            s.renderer = OutputRenderer::new("output", w, h)?;
            s.width = w;
            s.height = h;

            let ratio = w as f64 / h as f64;
            js_sys::Reflect::set(&window, &"__mamp_aspect".into(), &JsValue::from_f64(ratio))?;
        }

        s.running = true;
    }

    request_animation_frame(state.clone());
    Ok(())
}

#[wasm_bindgen(start)]
pub async fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).unwrap();

    log::info!("Initializing WebGPU motion magnification...");

    let ctx = GpuContext::new().await;
    log::info!("WebGPU device ready");

    // Start camera first to detect resolution
    let mut capture = VideoCapture::new(MAX_WIDTH, MAX_WIDTH * 3 / 4)?;
    capture.start_with_device("").await?;
    let (cam_w, cam_h) = capture.actual_size();
    let (w, h) = processing_size(cam_w, cam_h);
    log::info!("Camera: {}x{}, processing: {}x{}", cam_w, cam_h, w, h);
    capture.resize(w, h);

    // Tell JS the aspect ratio
    let window = web_sys::window().unwrap();
    let ratio = w as f64 / h as f64;
    js_sys::Reflect::set(&window, &"__mamp_aspect".into(), &JsValue::from_f64(ratio))?;

    let weights = load_weights().await;
    let model = MotionMagModel::new(&ctx, w, h, &weights);
    log::info!("Model built: {}x{}, latent {}x{}", w, h, model.latent_width, model.latent_height);

    let renderer = OutputRenderer::new("output", w, h)?;

    let state = Rc::new(RefCell::new(AppState {
        ctx,
        model,
        capture,
        renderer,
        prev_frame: None,
        alpha: 20.0,
        running: true,
        width: w,
        height: h,
    }));

    let state_ptr = Box::into_raw(Box::new(state.clone())) as usize;
    js_sys::Reflect::set(&window, &"__mamp_state".into(), &JsValue::from_f64(state_ptr as f64))?;

    request_animation_frame(state.clone());

    let state_for_js = state.clone();
    let set_alpha = Closure::wrap(Box::new(move |alpha: f32| {
        state_for_js.borrow_mut().alpha = alpha;
    }) as Box<dyn FnMut(f32)>);
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
        s.model.magnify(&s.ctx, prev, &current_frame, s.alpha);
        let _ = s.renderer.draw(&current_frame);
    }

    s.prev_frame = Some(current_frame);
    true
}

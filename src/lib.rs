// lib.rs — WASM entry point for real-time motion magnification

mod gpu;
#[allow(dead_code)]
mod model;
#[allow(dead_code)]
mod video;
mod weights;

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use gpu::GpuContext;
use model::{ModelWeights, MotionMagModel};
use video::{OutputRenderer, VideoCapture};

// Processing resolution caps per quality level.
// Must be divisible by 2 for the stride-2 encoder.
const QUALITY_MAX_WIDTHS: [u32; 3] = [480, 320, 240];

/// Adaptive frame-skip thresholds (milliseconds).
const THRESHOLD_EVERY_FRAME_MS: f64 = 20.0;
const THRESHOLD_SKIP_HALF_MS: f64 = 33.0;
const THRESHOLD_SKIP_THREE_QUARTER_MS: f64 = 66.0;

/// EMA smoothing factor for frame time tracking.
const FRAME_TIME_ALPHA: f64 = 0.1;

/// Double-buffered frame storage.  Two pre-allocated `Vec<u32>` buffers are
/// swapped each frame so that neither allocation nor cloning happens in the
/// hot loop.
struct FrameBuffers {
    pub bufs: [Vec<u32>; 2],
    /// Index into `bufs` for the *current* (most recently captured) frame.
    current: usize,
    /// Set once we have captured at least two frames (so prev is valid).
    has_prev: bool,
}

impl FrameBuffers {
    fn new(pixel_count: usize) -> Self {
        Self {
            bufs: [
                Vec::with_capacity(pixel_count),
                Vec::with_capacity(pixel_count),
            ],
            current: 0,
            has_prev: false,
        }
    }

    /// Advance to the next frame and return the buffer that should receive the
    /// new capture data.  After this call the old "current" buffer becomes the
    /// "previous" buffer.
    fn _advance(&mut self) -> &mut Vec<u32> {
        if !self.bufs[self.current].is_empty() {
            // We already have at least one captured frame; after swapping, the
            // old current is reachable as prev.
            self.current ^= 1;
            self.has_prev = true;
        }
        &mut self.bufs[self.current]
    }

    /// Like advance() but only updates the index, doesn't return a reference.
    fn advance_index(&mut self) {
        if !self.bufs[self.current].is_empty() {
            self.current ^= 1;
            self.has_prev = true;
        }
    }

    fn current_idx(&self) -> usize {
        self.current
    }

    fn current_frame(&self) -> &[u32] {
        &self.bufs[self.current]
    }

    fn prev_frame(&self) -> Option<&[u32]> {
        if self.has_prev {
            Some(&self.bufs[self.current ^ 1])
        } else {
            None
        }
    }

    /// Reset state (e.g. when switching cameras or changing resolution).
    fn reset(&mut self) {
        self.has_prev = false;
        self.bufs[0].clear();
        self.bufs[1].clear();
    }
}

/// State shared across animation frames.
struct AppState {
    ctx: GpuContext,
    model: MotionMagModel,
    capture: VideoCapture,
    renderer: OutputRenderer,
    frames: FrameBuffers,
    alpha: f32,
    running: bool,
    width: u32,
    height: u32,
    quality: u32,
    avg_frame_time_ms: f64,
    frame_index: u64,
    fps_frame_count: u32,
    fps_last_update: f64,
    /// Last magnified frame read back from GPU (displayed while next frame processes).
    magnified_pixels: Vec<u32>,
    /// True while a map_async is in-flight on buf_staging.
    mapping_pending: Rc<Cell<bool>>,
    /// True once we have at least one magnified frame to display.
    has_magnified: bool,
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

    // Fetch all 14 weight files in parallel (one round-trip instead of 14)
    let urls: Vec<String> = layer_specs.iter().map(|(name, _)| format!("weights/{}.bin", name)).collect();
    let loaded = match weights::load_all_parallel(&urls).await {
        Ok(data) => data,
        Err(e) => {
            log::warn!("Failed to load weights: {:?} — using random", e);
            return ModelWeights::random();
        }
    };

    // Validate sizes
    for (i, (name, expected_len)) in layer_specs.iter().enumerate() {
        if loaded[i].len() != *expected_len {
            log::warn!("{}: expected {} floats, got {} — using random", name, expected_len, loaded[i].len());
            return ModelWeights::random();
        }
    }
    log::info!("Loaded {} weight tensors in parallel", loaded.len());

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
/// Scales down to `max_width`, keeps aspect ratio, ensures even dimensions.
fn processing_size(cam_w: u32, cam_h: u32, max_width: u32) -> (u32, u32) {
    let (mut w, mut h) = if cam_w > max_width {
        let scale = max_width as f64 / cam_w as f64;
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

/// Return the current time in milliseconds via performance.now().
fn perf_now() -> f64 {
    web_sys::window()
        .unwrap()
        .performance()
        .unwrap()
        .now()
}

/// Determine how many out of every N frames should run magnify.
/// Returns (run_every_nth, period) — run magnify when frame_index % period < run_every_nth.
fn skip_policy(avg_ms: f64) -> (u64, u64) {
    if avg_ms < THRESHOLD_EVERY_FRAME_MS {
        // Fast enough: magnify every frame
        (1, 1)
    } else if avg_ms < THRESHOLD_SKIP_HALF_MS {
        // Moderate: still every frame (between 20-33ms is fine)
        (1, 1)
    } else if avg_ms < THRESHOLD_SKIP_THREE_QUARTER_MS {
        // Slow: skip magnify on alternating frames (run 1 out of 2)
        (1, 2)
    } else {
        // Very slow: run magnify 1 out of 4 frames
        (1, 4)
    }
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
        s.frames.reset();
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
        let max_w = QUALITY_MAX_WIDTHS[s.quality as usize];
        let (w, h) = processing_size(cam_w, cam_h, max_w);
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
    let initial_max = QUALITY_MAX_WIDTHS[0];
    let mut capture = VideoCapture::new(initial_max, initial_max * 3 / 4)?;
    capture.start_with_device("").await?;
    let (cam_w, cam_h) = capture.actual_size();
    let (w, h) = processing_size(cam_w, cam_h, initial_max);
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

    let now = perf_now();
    let state = Rc::new(RefCell::new(AppState {
        ctx,
        model,
        capture,
        renderer,
        frames: FrameBuffers::new((w * h) as usize),
        alpha: 20.0,
        running: true,
        width: w,
        height: h,
        quality: 0,
        avg_frame_time_ms: 0.0,
        frame_index: 0,
        fps_frame_count: 0,
        fps_last_update: now,
        magnified_pixels: vec![0u32; (w * h) as usize],
        mapping_pending: Rc::new(Cell::new(false)),
        has_magnified: false,
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

    let state_for_quality = state.clone();
    let set_quality = Closure::wrap(Box::new(move |level: u32| {
        let level = level.min(QUALITY_MAX_WIDTHS.len() as u32 - 1);
        let mut s = state_for_quality.borrow_mut();
        if s.quality == level {
            return;
        }
        s.quality = level;
        let max_w = QUALITY_MAX_WIDTHS[level as usize];
        let (cam_w, cam_h) = s.capture.actual_size();
        let (w, h) = processing_size(cam_w, cam_h, max_w);
        log::info!("Quality level {}: {}x{}", level, w, h);

        if w != s.width || h != s.height {
            s.capture.resize(w, h);
            let weights = ModelWeights::random();
            s.model = MotionMagModel::new(&s.ctx, w, h, &weights);
            // Rebuild the renderer; if it fails, keep the old dimensions.
            match OutputRenderer::new("output", w, h) {
                Ok(r) => {
                    s.renderer = r;
                    s.width = w;
                    s.height = h;
                    s.frames.reset();
                    // Reset frame-rate stats for the new resolution
                    s.avg_frame_time_ms = 0.0;
                    s.frame_index = 0;
                }
                Err(e) => {
                    log::error!("Failed to rebuild renderer: {:?}", e);
                }
            }
        }
    }) as Box<dyn FnMut(u32)>);
    js_sys::Reflect::set(&window, &"setQuality".into(), set_quality.as_ref())?;
    set_quality.forget();

    // Initialize __mamp_fps to 0
    js_sys::Reflect::set(&window, &"__mamp_fps".into(), &JsValue::from_f64(0.0))?;

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
    let t0 = perf_now();

    let mut s = state.borrow_mut();
    if !s.running {
        return false;
    }

    // Grab frame - clone to release the Ref borrow on s.capture
    let frame: Vec<u32> = match s.capture.grab_frame() {
        Ok(f) => f.clone(),
        Err(e) => {
            log::error!("Frame grab failed: {:?}", e);
            return true;
        }
    };
    s.frames.advance_index();
    let idx = s.frames.current_idx();
    let dest = &mut s.frames.bufs[idx];
    dest.clear();
    dest.extend_from_slice(&frame);

    // Read back magnified pixels if the previous map_async completed.
    // Must unmap BEFORE calling magnify() which writes to the staging buffer.
    if s.mapping_pending.get() {
        {
            let view = s.model.buf_staging.slice(..).get_mapped_range();
            let pixels: &[u32] = bytemuck::cast_slice(&view);
            s.magnified_pixels.clear();
            s.magnified_pixels.extend_from_slice(pixels);
            s.has_magnified = true;
        } // view dropped here, releasing the borrow on buf_staging
        s.model.buf_staging.unmap();
        s.mapping_pending.set(false);
    }

    // Run magnification if we have two frames and staging is free
    if let Some(prev) = s.frames.prev_frame() {
        let (run_count, period) = skip_policy(s.avg_frame_time_ms);
        let should_magnify = (s.frame_index % period) < run_count;

        if should_magnify && !s.mapping_pending.get() {
            let current = s.frames.current_frame();
            s.model.magnify(&s.ctx, prev, current, s.alpha);

            // Request async mapping of the staging buffer
            let pending = s.mapping_pending.clone();
            s.model.buf_staging.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    pending.set(true);
                }
            });
        }
    }

    // Display magnified frame if available, otherwise raw camera
    if s.has_magnified {
        let _ = s.renderer.draw(&s.magnified_pixels);
    } else {
        let _ = s.renderer.draw(s.frames.current_frame());
    }

    s.frame_index += 1;

    // Update EMA of frame time
    let elapsed = perf_now() - t0;
    if s.avg_frame_time_ms == 0.0 {
        s.avg_frame_time_ms = elapsed;
    } else {
        s.avg_frame_time_ms =
            FRAME_TIME_ALPHA * elapsed + (1.0 - FRAME_TIME_ALPHA) * s.avg_frame_time_ms;
    }

    // FPS counter: update once per second
    s.fps_frame_count += 1;
    let now = perf_now();
    let dt = now - s.fps_last_update;
    if dt >= 1000.0 {
        let fps = (s.fps_frame_count as f64 / dt) * 1000.0;
        s.fps_frame_count = 0;
        s.fps_last_update = now;

        // Expose to JS as window.__mamp_fps
        let window = web_sys::window().unwrap();
        let _ = js_sys::Reflect::set(
            &window,
            &"__mamp_fps".into(),
            &JsValue::from_f64(fps),
        );
    }

    true
}

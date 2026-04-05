// app.rs — WASM application entry point and animation loop
// Only compiled on wasm32 targets.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use crate::gpu::GpuContext;
use crate::steerable::SteerablePipeline;
use crate::video::VideoCapture;

const DEFAULT_MAX_WIDTH: u32 = 640;

thread_local! {
    static SHARED: RefCell<Option<Rc<Shared>>> = RefCell::new(None);
}

/// Max FFT length based on device's workgroup shared memory.
/// 2048 needs 16KB, 1024 needs 8KB.
fn max_fft_for_device(device: &wgpu::Device) -> u32 {
    let shared = device.limits().max_compute_workgroup_storage_size;
    // Each complex value = 2 × f32 = 8 bytes
    let max_n = shared / 8;
    if max_n >= 2048 { 2048 }
    else if max_n >= 1024 { 1024 }
    else { 512 }
}

struct Shared {
    state: RefCell<AppState>,
    running: Cell<bool>,
}

struct AppState {
    ctx: GpuContext,
    pipeline: SteerablePipeline,
    capture: VideoCapture,
    amplification: f32,
    freq_low: f32,
    freq_high: f32,
    n_scales: u32,
    n_orient: u32,
    width: u32,
    height: u32,
    cam_w: u32,
    cam_h: u32,
    active_device_id: String,
    active_facing: String,
}

fn processing_size(cam_w: u32, cam_h: u32, max_width: u32) -> (u32, u32) {
    let (mut w, mut h) = if cam_w > max_width {
        let scale = max_width as f64 / cam_w as f64;
        ((cam_w as f64 * scale) as u32, (cam_h as f64 * scale) as u32)
    } else {
        (cam_w, cam_h)
    };
    w &= !1;
    h &= !1;
    if w == 0 { w = 2; }
    if h == 0 { h = 2; }
    (w, h)
}

fn push_cam_w(cam_w: u32) {
    let window = web_sys::window().unwrap();
    let _ = js_sys::Reflect::set(&window, &"__mamp_cam_w".into(), &JsValue::from_f64(cam_w as f64));
}

fn perf_now() -> f64 {
    web_sys::window().unwrap().performance().unwrap().now()
}

async fn yield_to_browser() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        web_sys::window()
            .unwrap()
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0)
            .unwrap();
    });
    wasm_bindgen_futures::JsFuture::from(promise).await.unwrap();
}

#[allow(clippy::await_holding_refcell_ref)]
#[wasm_bindgen]
pub async fn start_with_camera(device_id: &str, facing_mode: &str) -> Result<(), JsValue> {
    let shared = SHARED.with(|s| s.borrow().clone())
        .ok_or_else(|| JsValue::from_str("Not initialized yet"))?;
    let window = web_sys::window().unwrap();

    shared.running.set(false);

    loop {
        yield_to_browser().await;
        if shared.state.try_borrow_mut().is_ok() { break; }
    }

    {
        let s = shared.state.borrow();
        s.capture.start_with_device(device_id, facing_mode, 0).await?;
    }

    {
        let mut s = shared.state.borrow_mut();
        s.active_device_id = device_id.to_string();
        s.active_facing = facing_mode.to_string();
        let (cam_w, cam_h) = s.capture.actual_size();
        s.cam_w = cam_w;
        s.cam_h = cam_h;
        push_cam_w(cam_w);
        let (w, h) = processing_size(cam_w, cam_h, DEFAULT_MAX_WIDTH);
        log::info!("Camera: {}x{}, processing: {}x{}", cam_w, cam_h, w, h);

        if w != s.width || h != s.height {
            s.capture.resize(w, h);
            s.pipeline = SteerablePipeline::new(&s.ctx, w, h, max_fft_for_device(&s.ctx.device), s.n_scales, s.n_orient);
            s.width = w;
            s.height = h;

            let ratio = w as f64 / h as f64;
            js_sys::Reflect::set(&window, &"__mamp_aspect".into(), &JsValue::from_f64(ratio))?;
        } else {
            // Same resolution, different camera — reset phase state
            s.pipeline.reset_state();
        }
        shared.running.set(true);
    }

    Ok(())
}

#[allow(clippy::await_holding_refcell_ref)]
#[wasm_bindgen]
pub async fn set_resolution(max_w: u32) -> Result<(), JsValue> {
    let shared = SHARED.with(|s| s.borrow().clone())
        .ok_or_else(|| JsValue::from_str("Not initialized yet"))?;

    shared.running.set(false);

    loop {
        yield_to_browser().await;
        if shared.state.try_borrow_mut().is_ok() { break; }
    }

    // Resize capture canvas to target, then restart camera so it provides
    // enough pixels. Must restart FIRST — cam_w/cam_h from a previous
    // low-res setting would make processing_size think nothing changed.
    let (device_id, facing) = {
        let mut s = shared.state.borrow_mut();
        // Set capture canvas to target so camera requests ideal:max_w
        let target_w = max_w.min(1280);
        let target_h = (target_w as u64 * 3 / 4) as u32; // rough 4:3 guess
        s.capture.resize(target_w, target_h);
        (s.active_device_id.clone(), s.active_facing.clone())
    };

    {
        let s = shared.state.borrow();
        s.capture.start_with_device(&device_id, &facing, max_w).await?;
    }

    {
        let mut s = shared.state.borrow_mut();
        let (cam_w, cam_h) = s.capture.actual_size();
        s.cam_w = cam_w;
        s.cam_h = cam_h;
        push_cam_w(cam_w);
        let (w, h) = processing_size(cam_w, cam_h, max_w);
        log::info!("Resolution: {}x{} (camera {}x{})", w, h, cam_w, cam_h);
        s.capture.resize(w, h);
        s.pipeline = SteerablePipeline::new(&s.ctx, w, h, max_fft_for_device(&s.ctx.device), s.n_scales, s.n_orient);
        s.width = w;
        s.height = h;
        let window = web_sys::window().unwrap();
        let ratio = w as f64 / h as f64;
        let _ = js_sys::Reflect::set(&window, &"__mamp_aspect".into(), &JsValue::from_f64(ratio));
        let _ = js_sys::Reflect::set(&window, &"__mamp_res".into(),
            &JsValue::from_str(&format!("{}x{} cam {}x{}", w, h, cam_w, cam_h)));
        shared.running.set(true);
    }

    Ok(())
}

#[wasm_bindgen(start)]
pub async fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).unwrap();

    log::info!("Initializing steerable pyramid...");

    let ctx = GpuContext::new().await;
    log::info!("WebGPU device ready");

    let initial_max = DEFAULT_MAX_WIDTH;
    let mut capture = VideoCapture::new(initial_max, initial_max * 3 / 4)?;
    capture.start_with_device("", "", 0).await?;
    let (cam_w, cam_h) = capture.actual_size();
    push_cam_w(cam_w);
    let (w, h) = processing_size(cam_w, cam_h, DEFAULT_MAX_WIDTH);
    log::info!("Camera: {}x{}, processing: {}x{}", cam_w, cam_h, w, h);
    capture.resize(w, h);

    let window = web_sys::window().unwrap();
    let ratio = w as f64 / h as f64;
    js_sys::Reflect::set(&window, &"__mamp_aspect".into(), &JsValue::from_f64(ratio))?;

    let n_scales = 3u32;
    let n_orient = 4u32;
    let pipeline = SteerablePipeline::new(&ctx, w, h, max_fft_for_device(&ctx.device), n_scales, n_orient);
    log::info!("Steerable pipeline ready: {}x{}", w, h);

    let shared = Rc::new(Shared {
        state: RefCell::new(AppState {
            ctx, pipeline, capture,
            amplification: 30.0, freq_low: 0.5, freq_high: 3.0, n_scales, n_orient,
            width: w, height: h, cam_w, cam_h,
            active_device_id: String::new(), active_facing: String::new(),
        }),
        running: Cell::new(true),
    });

    SHARED.with(|s| *s.borrow_mut() = Some(shared.clone()));

    wasm_bindgen_futures::spawn_local(run_loop(shared.clone()));

    let c1 = shared.clone();
    let set_amp = Closure::wrap(Box::new(move |v: f32| {
        if let Ok(mut s) = c1.state.try_borrow_mut() { s.amplification = v; }
    }) as Box<dyn FnMut(f32)>);
    js_sys::Reflect::set(&window, &"setMagnification".into(), set_amp.as_ref())?;
    set_amp.forget();

    let c2 = shared.clone();
    let set_fl = Closure::wrap(Box::new(move |v: f32| {
        if let Ok(mut s) = c2.state.try_borrow_mut() { s.freq_low = v; }
    }) as Box<dyn FnMut(f32)>);
    js_sys::Reflect::set(&window, &"setFreqLow".into(), set_fl.as_ref())?;
    set_fl.forget();

    let c3 = shared.clone();
    let set_fh = Closure::wrap(Box::new(move |v: f32| {
        if let Ok(mut s) = c3.state.try_borrow_mut() { s.freq_high = v; }
    }) as Box<dyn FnMut(f32)>);
    js_sys::Reflect::set(&window, &"setFreqHigh".into(), set_fh.as_ref())?;
    set_fh.forget();

    let c4 = shared.clone();
    let toggle = Closure::wrap(Box::new(move || {
        c4.running.set(!c4.running.get());
    }) as Box<dyn FnMut()>);
    js_sys::Reflect::set(&window, &"toggleMagnification".into(), toggle.as_ref())?;
    toggle.forget();

    // setResolution is now a wasm_bindgen async function (set_resolution),
    // called directly from JS — no closure needed.

    let c6 = shared.clone();
    let set_quality = Closure::wrap(Box::new(move |scales: u32, orient: u32| {
        if let Ok(mut s) = c6.state.try_borrow_mut() {
            if scales != s.n_scales || orient != s.n_orient {
                s.n_scales = scales.clamp(1, 4);
                s.n_orient = orient.clamp(1, 8);
                let max_fft = max_fft_for_device(&s.ctx.device);
                s.pipeline = SteerablePipeline::new(&s.ctx, s.width, s.height, max_fft, s.n_scales, s.n_orient);
            }
        }
    }) as Box<dyn FnMut(u32, u32)>);
    js_sys::Reflect::set(&window, &"setQuality".into(), set_quality.as_ref())?;
    set_quality.forget();

    js_sys::Reflect::set(&window, &"__mamp_fps".into(), &JsValue::from_f64(0.0))?;

    Ok(())
}

async fn next_frame() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        web_sys::window().unwrap().request_animation_frame(&resolve).unwrap();
    });
    wasm_bindgen_futures::JsFuture::from(promise).await.unwrap();
}

async fn run_loop(shared: Rc<Shared>) {
    let mut frame: Vec<u8> = Vec::new();
    let mut frame_count: u32 = 0;
    let mut last_fps_time = perf_now();
    let mut estimated_fps: f32 = 30.0;

    loop {
        next_frame().await;

        if !shared.running.get() {
            yield_to_browser().await;
            continue;
        }

        frame.clear();
        {
            let s = shared.state.borrow();
            if s.capture.grab_frame_into(&mut frame).is_err() { continue; }
        }

        {
            let mut s = shared.state.borrow_mut();
            let AppState { ref ctx, ref mut pipeline, amplification, freq_low, freq_high, .. } = *s;
            pipeline.process_and_render(ctx, &frame, amplification, freq_low, freq_high, estimated_fps);
        }

        frame_count += 1;
        let now = perf_now();
        if now - last_fps_time >= 1000.0 {
            let fps = (frame_count as f64 / (now - last_fps_time)) * 1000.0;
            estimated_fps = fps as f32;
            frame_count = 0;
            last_fps_time = now;
            let window = web_sys::window().unwrap();
            let _ = js_sys::Reflect::set(&window, &"__mamp_fps".into(), &JsValue::from_f64(fps));
            let (pw, ph, cw, ch) = {
                let s = shared.state.borrow();
                (s.width, s.height, s.cam_w, s.cam_h)
            };
            let _ = js_sys::Reflect::set(&window, &"__mamp_res".into(),
                &JsValue::from_str(&format!("{}x{} cam {}x{}", pw, ph, cw, ch)));
        }
    }
}

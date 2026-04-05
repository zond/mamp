// app.rs — WASM application entry point and animation loop
// Only compiled on wasm32 targets.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use crate::gpu::GpuContext;
use crate::steerable::SteerablePipeline;
use crate::video::VideoCapture;

const DEFAULT_MAX_WIDTH: u32 = 640;

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
    width: u32,
    height: u32,
    cam_w: u32,
    cam_h: u32,
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
pub async fn start_with_camera(device_id: &str) -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let state_js = js_sys::Reflect::get(&window, &"__mamp_state".into())?;
    if state_js.is_undefined() {
        return Err(JsValue::from_str("Not initialized yet"));
    }
    let ptr = state_js.as_f64().unwrap() as usize;
    let shared: &Rc<Shared> = unsafe { &*(ptr as *const Rc<Shared>) };

    shared.running.set(false);

    loop {
        yield_to_browser().await;
        if shared.state.try_borrow_mut().is_ok() { break; }
    }

    {
        let s = shared.state.borrow();
        s.capture.start_with_device(device_id).await?;
    }

    {
        let mut s = shared.state.borrow_mut();
        let (cam_w, cam_h) = s.capture.actual_size();
        let (w, h) = processing_size(cam_w, cam_h, DEFAULT_MAX_WIDTH);
        log::info!("Camera: {}x{}, processing: {}x{}", cam_w, cam_h, w, h);

        if w != s.width || h != s.height {
            s.capture.resize(w, h);
            s.pipeline = SteerablePipeline::new(&s.ctx, w, h, 1024);
            s.width = w;
            s.height = h;

            let ratio = w as f64 / h as f64;
            js_sys::Reflect::set(&window, &"__mamp_aspect".into(), &JsValue::from_f64(ratio))?;
        }
        shared.running.set(true);
    }

    Ok(())
}

#[wasm_bindgen(start)]
pub async fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).unwrap();

    log::info!("Initializing EVM motion magnification...");

    let ctx = GpuContext::new().await;
    log::info!("WebGPU device ready");

    let initial_max = DEFAULT_MAX_WIDTH;
    let mut capture = VideoCapture::new(initial_max, initial_max * 3 / 4)?;
    capture.start_with_device("").await?;
    let (cam_w, cam_h) = capture.actual_size();
    let (w, h) = processing_size(cam_w, cam_h, DEFAULT_MAX_WIDTH);
    log::info!("Camera: {}x{}, processing: {}x{}", cam_w, cam_h, w, h);
    capture.resize(w, h);

    let window = web_sys::window().unwrap();
    let ratio = w as f64 / h as f64;
    js_sys::Reflect::set(&window, &"__mamp_aspect".into(), &JsValue::from_f64(ratio))?;

    let pipeline = SteerablePipeline::new(&ctx, w, h, 1024);
    log::info!("Steerable pipeline ready: {}x{}", w, h);

    let shared = Rc::new(Shared {
        state: RefCell::new(AppState {
            ctx, pipeline, capture,
            amplification: 30.0, freq_low: 0.5, freq_high: 3.0,
            width: w, height: h, cam_w, cam_h,
        }),
        running: Cell::new(true),
    });

    let shared_ptr = Box::into_raw(Box::new(shared.clone())) as usize;
    js_sys::Reflect::set(&window, &"__mamp_state".into(), &JsValue::from_f64(shared_ptr as f64))?;

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

    let c5 = shared.clone();
    let set_res = Closure::wrap(Box::new(move |max_w: u32| {
        if let Ok(mut s) = c5.state.try_borrow_mut() { rebuild_pipeline(&mut s, max_w); }
    }) as Box<dyn FnMut(u32)>);
    js_sys::Reflect::set(&window, &"setResolution".into(), set_res.as_ref())?;
    set_res.forget();

    js_sys::Reflect::set(&window, &"__mamp_fps".into(), &JsValue::from_f64(0.0))?;

    Ok(())
}

async fn next_frame() {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        web_sys::window().unwrap().request_animation_frame(&resolve).unwrap();
    });
    wasm_bindgen_futures::JsFuture::from(promise).await.unwrap();
}

fn rebuild_pipeline(s: &mut AppState, max_w: u32) {
    let (w, h) = processing_size(s.cam_w, s.cam_h, max_w);
    if w == s.width && h == s.height { return; }
    log::info!("Resize: {}x{} -> {}x{}", s.width, s.height, w, h);
    s.capture.resize(w, h);
    s.pipeline = SteerablePipeline::new(&s.ctx, w, h, 1024);
    s.width = w;
    s.height = h;
}

async fn run_loop(shared: Rc<Shared>) {
    let mut frame: Vec<u32> = Vec::new();
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
            let (cw, ch) = { let s = shared.state.borrow(); (s.width, s.height) };
            let _ = js_sys::Reflect::set(&window, &"__mamp_res".into(),
                &JsValue::from_str(&format!("{}x{}", cw, ch)));
        }
    }
}

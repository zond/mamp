// lib.rs — WASM entry point for Eulerian Video Magnification

mod evm;
mod gpu;
mod video;

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;

use evm::EvmPipeline;
use gpu::GpuContext;
use video::VideoCapture;

const DEFAULT_MAX_WIDTH: u32 = 640;

struct AppState {
    ctx: GpuContext,
    evm: EvmPipeline,
    capture: VideoCapture,
    amplification: f32,
    freq_low: f32,
    freq_high: f32,
    running: bool,
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
        web_sys::window().unwrap()
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 0).unwrap();
    });
    wasm_bindgen_futures::JsFuture::from(promise).await.unwrap();
}

fn get_canvas() -> web_sys::HtmlCanvasElement {
    web_sys::window().unwrap()
        .document().unwrap()
        .get_element_by_id("output").unwrap()
        .dyn_into::<web_sys::HtmlCanvasElement>().unwrap()
}

#[wasm_bindgen]
pub async fn start_with_camera(device_id: &str) -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let state_js = js_sys::Reflect::get(&window, &"__mamp_state".into())?;
    if state_js.is_undefined() {
        return Err(JsValue::from_str("Not initialized yet"));
    }
    let ptr = state_js.as_f64().unwrap() as usize;
    let state: &Rc<RefCell<AppState>> = unsafe { &*(ptr as *const Rc<RefCell<AppState>>) };

    {
        let mut s = state.borrow_mut();
        s.running = false;
    }
    // Wait until run_loop actually yields (it checks running each frame).
    // Need enough yields for the current frame to finish processing.
    for _ in 0..10 {
        yield_to_browser().await;
        if state.try_borrow().is_ok() { break; }
    }

    state.borrow().capture.start_with_device(device_id).await?;

    {
        let mut s = state.borrow_mut();
        let (cam_w, cam_h) = s.capture.actual_size();
        let (w, h) = processing_size(cam_w, cam_h, DEFAULT_MAX_WIDTH);
        log::info!("Camera: {}x{}, processing: {}x{}", cam_w, cam_h, w, h);

        if w != s.width || h != s.height {
            s.capture.resize(w, h);
            let canvas = get_canvas();
            canvas.set_width(w);
            canvas.set_height(h);
            s.evm = EvmPipeline::new(&s.ctx, w, h);
            s.width = w;
            s.height = h;

            let ratio = w as f64 / h as f64;
            js_sys::Reflect::set(&window, &"__mamp_aspect".into(), &JsValue::from_f64(ratio))?;
        }
        s.running = true;
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

    let canvas = get_canvas();
    canvas.set_width(w);
    canvas.set_height(h);
    let evm = EvmPipeline::new(&ctx, w, h);
    log::info!("EVM pipeline ready: {}x{}", w, h);

    let state = Rc::new(RefCell::new(AppState {
        ctx, evm, capture,
        amplification: 30.0,
        freq_low: 0.5,
        freq_high: 3.0,
        running: true,
        width: w, height: h,
        cam_w, cam_h,
    }));

    let state_ptr = Box::into_raw(Box::new(state.clone())) as usize;
    js_sys::Reflect::set(&window, &"__mamp_state".into(), &JsValue::from_f64(state_ptr as f64))?;

    wasm_bindgen_futures::spawn_local(run_loop(state.clone()));

    // Expose controls to JS
    let s1 = state.clone();
    let set_amp = Closure::wrap(Box::new(move |v: f32| { s1.borrow_mut().amplification = v; }) as Box<dyn FnMut(f32)>);
    js_sys::Reflect::set(&window, &"setMagnification".into(), set_amp.as_ref())?;
    set_amp.forget();

    let s2 = state.clone();
    let set_fl = Closure::wrap(Box::new(move |v: f32| { s2.borrow_mut().freq_low = v; }) as Box<dyn FnMut(f32)>);
    js_sys::Reflect::set(&window, &"setFreqLow".into(), set_fl.as_ref())?;
    set_fl.forget();

    let s3 = state.clone();
    let set_fh = Closure::wrap(Box::new(move |v: f32| { s3.borrow_mut().freq_high = v; }) as Box<dyn FnMut(f32)>);
    js_sys::Reflect::set(&window, &"setFreqHigh".into(), set_fh.as_ref())?;
    set_fh.forget();

    let s4 = state.clone();
    let toggle = Closure::wrap(Box::new(move || { s4.borrow_mut().running = !s4.borrow().running; }) as Box<dyn FnMut()>);
    js_sys::Reflect::set(&window, &"toggleMagnification".into(), toggle.as_ref())?;
    toggle.forget();

    let s5 = state.clone();
    let set_res = Closure::wrap(Box::new(move |max_w: u32| {
        let mut s = s5.borrow_mut();
        rebuild_evm(&mut s, max_w);
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

fn rebuild_evm(s: &mut AppState, max_w: u32) {
    let (w, h) = processing_size(s.cam_w, s.cam_h, max_w);
    if w == s.width && h == s.height { return; }
    log::info!("Resize: {}x{} -> {}x{}", s.width, s.height, w, h);
    s.capture.resize(w, h);
    let canvas = get_canvas();
    canvas.set_width(w);
    canvas.set_height(h);
    s.evm = EvmPipeline::new(&s.ctx, w, h);
    s.width = w;
    s.height = h;
}

async fn run_loop(state: Rc<RefCell<AppState>>) {
    let mut frame: Vec<u32> = Vec::new();
    let mut frame_count: u32 = 0;
    let mut last_fps_time = perf_now();
    let mut estimated_fps: f32 = 30.0;

    loop {
        next_frame().await;

        let running = state.borrow().running;
        if !running {
            yield_to_browser().await;
            continue;
        }

        frame.clear();
        {
            let s = state.borrow();
            if s.capture.grab_frame_into(&mut frame).is_err() {
                continue;
            }
        }

        {
            let s = state.borrow();
            s.evm.process_and_render(&s.ctx, &frame, s.amplification, s.freq_low, s.freq_high, estimated_fps);
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
            let (cw, ch) = { let s = state.borrow(); (s.width, s.height) };
            let _ = js_sys::Reflect::set(&window, &"__mamp_res".into(),
                &JsValue::from_str(&format!("{}x{}", cw, ch)));
        }
    }
}

// lib.rs — WASM entry point for Eulerian Video Magnification

mod evm;
mod gpu;
mod video;

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use wasm_bindgen::prelude::*;

use evm::EvmPipeline;
use gpu::GpuContext;
use video::{OutputRenderer, VideoCapture};

const MAX_WIDTH: u32 = 480;
const FRAME_TIME_ALPHA: f64 = 0.1;

struct AppState {
    ctx: GpuContext,
    evm: EvmPipeline,
    capture: VideoCapture,
    renderer: OutputRenderer,
    amplification: f32,
    freq_low: f32,
    freq_high: f32,
    running: bool,
    width: u32,
    height: u32,
    generation: Rc<Cell<u32>>,
}

fn processing_size(cam_w: u32, cam_h: u32) -> (u32, u32) {
    let (mut w, mut h) = if cam_w > MAX_WIDTH {
        let scale = MAX_WIDTH as f64 / cam_w as f64;
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

    {
        let mut s = state.borrow_mut();
        s.running = false;
        s.generation.set(s.generation.get() + 1);
    }

    yield_to_browser().await;
    yield_to_browser().await;

    state.borrow().capture.start_with_device(device_id).await?;

    {
        let mut s = state.borrow_mut();
        let (cam_w, cam_h) = s.capture.actual_size();
        let (w, h) = processing_size(cam_w, cam_h);
        log::info!("Camera: {}x{}, processing: {}x{}", cam_w, cam_h, w, h);

        if w != s.width || h != s.height {
            s.capture.resize(w, h);
            s.evm = EvmPipeline::new(&s.ctx, w, h);
            s.renderer = OutputRenderer::new("output", w, h)?;
            s.width = w;
            s.height = h;

            let ratio = w as f64 / h as f64;
            js_sys::Reflect::set(&window, &"__mamp_aspect".into(), &JsValue::from_f64(ratio))?;
        }

        s.running = true;
    }

    // Don't spawn a new run_loop — the existing one is still alive,
    // polling s.running. It will resume on the next iteration.
    Ok(())
}

#[wasm_bindgen(start)]
pub async fn start() -> Result<(), JsValue> {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).unwrap();

    log::info!("Initializing EVM motion magnification...");

    let ctx = GpuContext::new().await;
    log::info!("WebGPU device ready");

    // Start camera to detect resolution
    let mut capture = VideoCapture::new(MAX_WIDTH, MAX_WIDTH * 3 / 4)?;
    capture.start_with_device("").await?;
    let (cam_w, cam_h) = capture.actual_size();
    let (w, h) = processing_size(cam_w, cam_h);
    log::info!("Camera: {}x{}, processing: {}x{}", cam_w, cam_h, w, h);
    capture.resize(w, h);

    let window = web_sys::window().unwrap();
    let ratio = w as f64 / h as f64;
    js_sys::Reflect::set(&window, &"__mamp_aspect".into(), &JsValue::from_f64(ratio))?;

    let evm = EvmPipeline::new(&ctx, w, h);
    log::info!("EVM pipeline ready: {}x{}", w, h);

    let renderer = OutputRenderer::new("output", w, h)?;

    let state = Rc::new(RefCell::new(AppState {
        ctx,
        evm,
        capture,
        renderer,
        amplification: 20.0,
        freq_low: 0.5,
        freq_high: 3.0,
        running: true,
        width: w,
        height: h,
        generation: Rc::new(Cell::new(0)),
    }));

    let state_ptr = Box::into_raw(Box::new(state.clone())) as usize;
    js_sys::Reflect::set(&window, &"__mamp_state".into(), &JsValue::from_f64(state_ptr as f64))?;

    wasm_bindgen_futures::spawn_local(run_loop(state.clone()));

    // Expose controls to JS
    let s1 = state.clone();
    let set_amp = Closure::wrap(Box::new(move |v: f32| {
        s1.borrow_mut().amplification = v;
    }) as Box<dyn FnMut(f32)>);
    js_sys::Reflect::set(&window, &"setMagnification".into(), set_amp.as_ref())?;
    set_amp.forget();

    let s2 = state.clone();
    let set_freq_low = Closure::wrap(Box::new(move |v: f32| {
        s2.borrow_mut().freq_low = v;
    }) as Box<dyn FnMut(f32)>);
    js_sys::Reflect::set(&window, &"setFreqLow".into(), set_freq_low.as_ref())?;
    set_freq_low.forget();

    let s3 = state.clone();
    let set_freq_high = Closure::wrap(Box::new(move |v: f32| {
        s3.borrow_mut().freq_high = v;
    }) as Box<dyn FnMut(f32)>);
    js_sys::Reflect::set(&window, &"setFreqHigh".into(), set_freq_high.as_ref())?;
    set_freq_high.forget();

    let s4 = state.clone();
    let toggle = Closure::wrap(Box::new(move || {
        let mut s = s4.borrow_mut();
        s.running = !s.running;
    }) as Box<dyn FnMut()>);
    js_sys::Reflect::set(&window, &"toggleMagnification".into(), toggle.as_ref())?;
    toggle.forget();

    js_sys::Reflect::set(&window, &"__mamp_fps".into(), &JsValue::from_f64(0.0))?;

    Ok(())
}

async fn next_frame() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        web_sys::window()
            .unwrap()
            .request_animation_frame(&resolve)
            .unwrap();
    });
    wasm_bindgen_futures::JsFuture::from(promise).await.unwrap();
}

async fn run_loop(state: Rc<RefCell<AppState>>) {
    let mut magnified_pixels: Vec<u32> = Vec::new();
    let mut has_magnified = false;
    let mut frame_count: u32 = 0;
    let mut last_fps_time = perf_now();
    let mut frame: Vec<u32> = Vec::new();
    let mut avg_frame_time: f64 = 0.0;
    let mut estimated_fps: f32 = 30.0;

    let map_ready = Rc::new(Cell::new(false));
    let mut map_pending = false;
    let mut map_generation: u32 = 0;

    loop {
        next_frame().await;
        let t0 = perf_now();

        let running = state.borrow().running;
        if !running {
            yield_to_browser().await;
            continue;
        }

        // Check if previous readback completed
        let cur_gen = state.borrow().generation.get();
        if map_pending && map_ready.get() {
            if map_generation == cur_gen {
                let s = state.borrow();
                let view = s.evm.buf_staging.slice(..).get_mapped_range();
                let pixels: &[u32] = bytemuck::cast_slice(&view);
                magnified_pixels.clear();
                magnified_pixels.extend_from_slice(pixels);
                has_magnified = true;
                drop(view);
                s.evm.buf_staging.unmap();
            }
            map_pending = false;
            map_ready.set(false);
        } else if map_pending && map_generation != cur_gen {
            map_pending = false;
            map_ready.set(false);
            has_magnified = false;
        }

        // Grab camera frame
        frame.clear();
        {
            let s = state.borrow();
            if s.capture.grab_frame_into(&mut frame).is_err() {
                continue;
            }
        }

        // Run EVM if no readback pending
        if !map_pending {
            let (amp, fl, fh) = {
                let s = state.borrow();
                (s.amplification, s.freq_low, s.freq_high)
            };

            {
                let s = state.borrow();
                s.evm.process_frame(&s.ctx, &frame, amp, fl, fh, estimated_fps);
            }

            // Request readback
            {
                let s = state.borrow();
                let flag = map_ready.clone();
                s.evm.buf_staging.slice(..).map_async(
                    wgpu::MapMode::Read,
                    move |r| { if r.is_ok() { flag.set(true); } },
                );
            }
            map_pending = true;
            map_generation = cur_gen;
        }

        // Display
        {
            let s = state.borrow();
            if has_magnified {
                let _ = s.renderer.draw(&magnified_pixels);
            } else {
                let _ = s.renderer.draw(&frame);
            }
        }

        // FPS tracking
        frame_count += 1;
        let elapsed = perf_now() - t0;
        avg_frame_time = if avg_frame_time == 0.0 { elapsed }
            else { FRAME_TIME_ALPHA * elapsed + (1.0 - FRAME_TIME_ALPHA) * avg_frame_time };

        let now = perf_now();
        if now - last_fps_time >= 1000.0 {
            let fps = (frame_count as f64 / (now - last_fps_time)) * 1000.0;
            estimated_fps = fps as f32;
            frame_count = 0;
            last_fps_time = now;
            let window = web_sys::window().unwrap();
            let _ = js_sys::Reflect::set(&window, &"__mamp_fps".into(), &JsValue::from_f64(fps));
        }
    }
}

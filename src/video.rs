// video.rs — Browser camera capture and canvas rendering via web-sys

use std::cell::Cell;
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    CanvasRenderingContext2d, HtmlCanvasElement, HtmlVideoElement, MediaStreamConstraints,
};

/// Manages camera capture and frame extraction.
pub struct VideoCapture {
    video: HtmlVideoElement,
    _canvas: HtmlCanvasElement,
    ctx2d: CanvasRenderingContext2d,
    pub width: u32,
    pub height: u32,
    mirrored: Cell<bool>,
}

impl VideoCapture {
    pub fn new(width: u32, height: u32) -> Result<Self, JsValue> {
        let document = web_sys::window().unwrap().document().unwrap();

        let video = document
            .create_element("video")?
            .dyn_into::<HtmlVideoElement>()?;
        video.set_attribute("autoplay", "")?;
        video.set_attribute("playsinline", "")?;
        video.set_width(width);
        video.set_height(height);
        video.set_attribute("style", "display:none")?;
        document.body().unwrap().append_child(&video)?;

        let canvas = document
            .create_element("canvas")?
            .dyn_into::<HtmlCanvasElement>()?;
        canvas.set_width(width);
        canvas.set_height(height);
        canvas.set_attribute("style", "display:none")?;
        document.body().unwrap().append_child(&canvas)?;

        let ctx_opts = js_sys::Object::new();
        js_sys::Reflect::set(&ctx_opts, &"willReadFrequently".into(), &JsValue::TRUE)?;
        let ctx2d = canvas
            .get_context_with_context_options("2d", &ctx_opts)?
            .unwrap()
            .dyn_into::<CanvasRenderingContext2d>()?;

        Ok(Self {
            video,
            _canvas: canvas,
            ctx2d,
            width,
            height,
            mirrored: Cell::new(true), // assume front-facing until proven otherwise
        })
    }

    /// Start camera with optional device ID (empty string = default camera).
    pub async fn start_with_device(&self, device_id: &str) -> Result<(), JsValue> {
        // Fully release old camera first
        if let Some(old_stream) = self.video.src_object() {
            self.video.pause().ok();
            let old: web_sys::MediaStream = old_stream.unchecked_into();
            let tracks = old.get_tracks();
            let n = tracks.length();
            for i in 0..n {
                let track: web_sys::MediaStreamTrack = tracks.get(i).unchecked_into();
                let state_before = js_sys::Reflect::get(track.as_ref(), &"readyState".into())
                    .ok().and_then(|v| v.as_string()).unwrap_or_default();
                log::info!("Stopping track: {} state={}", track.label(), state_before);
                track.stop();
                let state_after = js_sys::Reflect::get(track.as_ref(), &"readyState".into())
                    .ok().and_then(|v| v.as_string()).unwrap_or_default();
                log::info!("Track stopped: state={}", state_after);
            }
            self.video.set_src_object(None);
            self.video.load();
            log::info!("Old stream released ({} tracks stopped)", n);
        } else {
            log::info!("No old stream to release");
        }

        let window = web_sys::window().unwrap();
        let navigator = window.navigator();
        let media_devices = navigator.media_devices()?;

        let constraints = MediaStreamConstraints::new();
        let video_constraints = js_sys::Object::new();
        if !device_id.is_empty() {
            js_sys::Reflect::set(&video_constraints, &"deviceId".into(),
                &JsValue::from_str(device_id))?;
        }
        let ideal_w = js_sys::Object::new();
        js_sys::Reflect::set(&ideal_w, &"ideal".into(), &JsValue::from(640))?;
        js_sys::Reflect::set(&video_constraints, &"width".into(), &ideal_w)?;
        constraints.set_video(&video_constraints.into());
        constraints.set_audio(&JsValue::FALSE);

        // Retry with long delays — some devices need seconds to release camera hardware
        log::info!("Requesting camera: deviceId={}", if device_id.is_empty() { "(default)" } else { device_id });
        let mut stream = None;
        let delays_ms = [500, 1500, 3000, 5000];
        for (attempt, &ms) in delays_ms.iter().enumerate() {
            log::info!("Waiting {}ms before attempt {}...", ms, attempt + 1);
            let delay = js_sys::Promise::new(&mut |resolve, _| {
                web_sys::window().unwrap()
                    .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms)
                    .unwrap();
            });
            wasm_bindgen_futures::JsFuture::from(delay).await?;

            let promise = media_devices.get_user_media_with_constraints(&constraints)?;
            match wasm_bindgen_futures::JsFuture::from(promise).await {
                Ok(s) => {
                    log::info!("getUserMedia succeeded on attempt {}", attempt + 1);
                    stream = Some(s);
                    break;
                }
                Err(e) if attempt < delays_ms.len() - 1 => {
                    let msg = format!("{:?}", e);
                    log::warn!("getUserMedia attempt {} failed: {}", attempt + 1, msg);
                }
                Err(e) => return Err(e),
            }
        }
        let stream = stream.unwrap();

        let stream_obj: web_sys::MediaStream = stream.unchecked_ref::<web_sys::MediaStream>().clone();
        self.video.set_src_object(Some(&stream_obj));

        // Detect facing mode to decide mirroring
        let tracks = stream_obj.get_video_tracks();
        if tracks.length() > 0 {
            let track: web_sys::MediaStreamTrack = tracks.get(0).unchecked_into();
            let settings = track.get_settings();
            let facing = js_sys::Reflect::get(settings.as_ref(), &"facingMode".into())
                .ok()
                .and_then(|v| v.as_string());
            // Mirror unless explicitly "environment" (back camera)
            self.mirrored.set(facing.as_deref() != Some("environment"));
        }

        let play_promise = self.video.play()?;
        wasm_bindgen_futures::JsFuture::from(play_promise).await?;

        // Wait a moment for the video to report its actual dimensions
        let promise = js_sys::Promise::new(&mut |resolve, _| {
            let window = web_sys::window().unwrap();
            window
                .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 200)
                .unwrap();
        });
        wasm_bindgen_futures::JsFuture::from(promise).await?;

        Ok(())
    }

    /// Get the camera's actual resolution (may differ from requested).
    pub fn actual_size(&self) -> (u32, u32) {
        let vw = self.video.video_width();
        let vh = self.video.video_height();
        if vw > 0 && vh > 0 {
            (vw, vh)
        } else {
            (self.width, self.height)
        }
    }

    /// Resize internal capture canvas to match new dimensions.
    pub fn resize(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
        self._canvas.set_width(width);
        self._canvas.set_height(height);
    }

    /// Grab the current video frame into a caller-provided buffer, reusing its
    /// allocation.  The buffer is cleared and refilled each call.
    pub fn grab_frame_into(&self, dest: &mut Vec<u32>) -> Result<(), JsValue> {
        if self.mirrored.get() {
            self.ctx2d.save();
            self.ctx2d.translate(self.width as f64, 0.0)?;
            self.ctx2d.scale(-1.0, 1.0)?;
        }

        self.ctx2d
            .draw_image_with_html_video_element_and_dw_and_dh(
                &self.video,
                0.0,
                0.0,
                self.width as f64,
                self.height as f64,
            )?;

        if self.mirrored.get() {
            self.ctx2d.restore();
        }

        let image_data =
            self.ctx2d
                .get_image_data(0.0, 0.0, self.width as f64, self.height as f64)?;

        let raw: Vec<u8> = image_data.data().0;

        dest.clear();
        dest.extend(raw.chunks_exact(4).map(|c| {
            (c[0] as u32) | ((c[1] as u32) << 8) | ((c[2] as u32) << 16) | ((c[3] as u32) << 24)
        }));

        Ok(())
    }
}

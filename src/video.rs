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
            mirrored: Cell::new(true),
        })
    }

    /// Try to acquire a media stream with the given constraints.
    /// Returns Ok(stream) on success, Err on failure.
    async fn try_get_stream(
        media_devices: &web_sys::MediaDevices,
        video_constraints: &JsValue,
    ) -> Result<web_sys::MediaStream, JsValue> {
        let constraints = MediaStreamConstraints::new();
        constraints.set_video(video_constraints);
        constraints.set_audio(&JsValue::FALSE);
        let stream_promise = media_devices.get_user_media_with_constraints(&constraints)?;
        let stream = wasm_bindgen_futures::JsFuture::from(stream_promise).await?;
        Ok(stream.unchecked_into())
    }

    /// Start camera with optional device ID and facing mode.
    /// Start camera. `target_width`: 0 = camera picks best, >0 = ideal hint.
    pub async fn start_with_device(&self, device_id: &str, facing_mode: &str, target_width: u32) -> Result<(), JsValue> {
        // Stop any existing stream
        let switching = if let Some(old_stream) = self.video.src_object() {
            let old: web_sys::MediaStream = old_stream.unchecked_into();
            let tracks = old.get_tracks();
            for i in 0..tracks.length() {
                let track: web_sys::MediaStreamTrack = tracks.get(i).unchecked_into();
                track.stop();
            }
            self.video.set_src_object(None);
            true
        } else {
            false
        };

        // track.stop() returns synchronously but hardware release is async.
        // Android Camera2 needs time to close the device.
        if switching {
            let delay = js_sys::Promise::new(&mut |resolve, _| {
                web_sys::window().unwrap()
                    .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 300)
                    .unwrap();
            });
            wasm_bindgen_futures::JsFuture::from(delay).await?;
        }

        let window = web_sys::window().unwrap();
        let media_devices = window.navigator().media_devices()?;

        // Build the fallback chain of constraint objects to try in order.
        let mut attempts: Vec<(String, JsValue)> = Vec::new();

        if !device_id.is_empty() {
            let vc = js_sys::Object::new();
            let exact = js_sys::Object::new();
            js_sys::Reflect::set(&exact, &"exact".into(), &JsValue::from_str(device_id))?;
            js_sys::Reflect::set(&vc, &"deviceId".into(), &exact)?;
            {
                let w_hint = if target_width > 0 { target_width } else { 4096 };
                let ideal_w = js_sys::Object::new();
                js_sys::Reflect::set(&ideal_w, &"ideal".into(), &JsValue::from(w_hint))?;
                js_sys::Reflect::set(&vc, &"width".into(), &ideal_w)?;
            }
            attempts.push((
                format!("deviceId={} w={}", device_id, if target_width > 0 { target_width } else { 4096 }),
                vc.into(),
            ));
        }

        if !facing_mode.is_empty() {
            // Attempt 2: facingMode alone
            let vc = js_sys::Object::new();
            let exact = js_sys::Object::new();
            js_sys::Reflect::set(&exact, &"exact".into(), &JsValue::from_str(facing_mode))?;
            js_sys::Reflect::set(&vc, &"facingMode".into(), &exact)?;
            attempts.push((
                format!("facingMode={}", facing_mode),
                vc.into(),
            ));
        }

        if !device_id.is_empty() {
            // Attempt 3: deviceId alone (no resolution constraints)
            let vc = js_sys::Object::new();
            let exact = js_sys::Object::new();
            js_sys::Reflect::set(&exact, &"exact".into(), &JsValue::from_str(device_id))?;
            js_sys::Reflect::set(&vc, &"deviceId".into(), &exact)?;
            attempts.push((
                format!("deviceId={} (no resolution)", device_id),
                vc.into(),
            ));
        }

        if attempts.is_empty() {
            // No device_id or facing_mode — request default camera
            attempts.push((
                "default camera".to_string(),
                JsValue::TRUE,
            ));
        }

        let mut last_err: Option<JsValue> = None;
        let mut stream_obj: Option<web_sys::MediaStream> = None;

        for (label, vc) in &attempts {
            log::info!("Trying camera: {}", label);
            match Self::try_get_stream(&media_devices, vc).await {
                Ok(s) => {
                    log::info!("Camera opened: {}", label);
                    stream_obj = Some(s);
                    break;
                }
                Err(e) => {
                    log::warn!("Camera attempt failed ({}): {:?}", label, e);
                    last_err = Some(e);
                }
            }
        }

        let stream_obj = match stream_obj {
            Some(s) => s,
            None => return Err(last_err.unwrap_or_else(|| JsValue::from_str("No camera available"))),
        };

        self.video.set_src_object(Some(&stream_obj));

        // Detect facing mode to decide mirroring
        let tracks = stream_obj.get_video_tracks();
        if tracks.length() > 0 {
            let track: web_sys::MediaStreamTrack = tracks.get(0).unchecked_into();
            let settings = track.get_settings();
            let facing = js_sys::Reflect::get(settings.as_ref(), &"facingMode".into())
                .ok()
                .and_then(|v| v.as_string());
            self.mirrored.set(facing.as_deref() != Some("environment"));
        }

        let play_promise = self.video.play()?;
        wasm_bindgen_futures::JsFuture::from(play_promise).await?;

        // Wait for video to report actual dimensions
        let promise = js_sys::Promise::new(&mut |resolve, _| {
            web_sys::window().unwrap()
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

    /// Grab the current video frame as raw RGBA bytes.
    pub fn grab_frame_into(&self, dest: &mut Vec<u8>) -> Result<(), JsValue> {
        if self.mirrored.get() {
            self.ctx2d.save();
            self.ctx2d.translate(self.width as f64, 0.0)?;
            self.ctx2d.scale(-1.0, 1.0)?;
        }

        self.ctx2d
            .draw_image_with_html_video_element_and_dw_and_dh(
                &self.video, 0.0, 0.0, self.width as f64, self.height as f64,
            )?;

        if self.mirrored.get() {
            self.ctx2d.restore();
        }

        let image_data =
            self.ctx2d
                .get_image_data(0.0, 0.0, self.width as f64, self.height as f64)?;

        // On little-endian WASM, RGBA bytes are already in u32 layout.
        // Return raw bytes — caller passes directly to write_buffer.
        *dest = image_data.data().0;

        Ok(())
    }
}

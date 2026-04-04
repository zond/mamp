// video.rs — Browser camera capture and canvas rendering via web-sys

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_sys::{
    CanvasRenderingContext2d, HtmlCanvasElement, HtmlVideoElement,
    MediaStreamConstraints, ImageData,
};

/// Manages camera capture and frame extraction.
pub struct VideoCapture {
    video: HtmlVideoElement,
    canvas: HtmlCanvasElement,
    ctx2d: CanvasRenderingContext2d,
    pub width: u32,
    pub height: u32,
}

impl VideoCapture {
    /// Create a video capture at the given resolution.
    pub fn new(width: u32, height: u32) -> Result<Self, JsValue> {
        let document = web_sys::window().unwrap().document().unwrap();

        // Hidden video element for camera stream
        let video = document
            .create_element("video")?
            .dyn_into::<HtmlVideoElement>()?;
        video.set_attribute("autoplay", "")?;
        video.set_attribute("playsinline", "")?;
        video.set_width(width);
        video.set_height(height);
        video.set_attribute("style", "display:none")?;
        document.body().unwrap().append_child(&video)?;

        // Offscreen canvas for pixel extraction
        let canvas = document
            .create_element("canvas")?
            .dyn_into::<HtmlCanvasElement>()?;
        canvas.set_width(width);
        canvas.set_height(height);
        canvas.set_attribute("style", "display:none")?;
        document.body().unwrap().append_child(&canvas)?;

        let ctx2d = canvas
            .get_context("2d")?
            .unwrap()
            .dyn_into::<CanvasRenderingContext2d>()?;

        Ok(Self { video, canvas, ctx2d, width, height })
    }

    /// Request camera access and start the stream.
    pub async fn start(&self) -> Result<(), JsValue> {
        let window = web_sys::window().unwrap();
        let navigator = window.navigator();
        let media_devices = navigator.media_devices()?;

        let mut constraints = MediaStreamConstraints::new();
        // Build video constraints with resolution
        let video_constraints = js_sys::Object::new();
        js_sys::Reflect::set(
            &video_constraints,
            &"width".into(),
            &JsValue::from(self.width),
        )?;
        js_sys::Reflect::set(
            &video_constraints,
            &"height".into(),
            &JsValue::from(self.height),
        )?;
        constraints.video(&video_constraints.into());
        constraints.audio(&JsValue::FALSE);

        let stream_promise = media_devices.get_user_media_with_constraints(&constraints)?;
        let stream = wasm_bindgen_futures::JsFuture::from(stream_promise).await?;

        self.video
            .set_src_object(Some(&stream.unchecked_into::<web_sys::MediaStream>()));

        // Wait for video to be ready
        let video = self.video.clone();
        let play_promise = video.play()?;
        wasm_bindgen_futures::JsFuture::from(play_promise).await?;

        Ok(())
    }

    /// Grab the current frame as packed RGBA u32 values.
    pub fn grab_frame(&self) -> Result<Vec<u32>, JsValue> {
        self.ctx2d.draw_image_with_html_video_element_and_dw_and_dh(
            &self.video,
            0.0, 0.0,
            self.width as f64,
            self.height as f64,
        )?;

        let image_data = self.ctx2d.get_image_data(
            0.0, 0.0,
            self.width as f64,
            self.height as f64,
        )?;

        let raw: Vec<u8> = image_data.data().0;

        // Pack RGBA bytes into u32
        let pixels: Vec<u32> = raw
            .chunks_exact(4)
            .map(|c| {
                (c[0] as u32)
                    | ((c[1] as u32) << 8)
                    | ((c[2] as u32) << 16)
                    | ((c[3] as u32) << 24)
            })
            .collect();

        Ok(pixels)
    }
}

/// Render output: write RGBA u32 data to a visible canvas.
pub struct OutputRenderer {
    canvas: HtmlCanvasElement,
    ctx2d: CanvasRenderingContext2d,
    width: u32,
    height: u32,
}

impl OutputRenderer {
    pub fn new(canvas_id: &str, width: u32, height: u32) -> Result<Self, JsValue> {
        let document = web_sys::window().unwrap().document().unwrap();
        let canvas = document
            .get_element_by_id(canvas_id)
            .ok_or("Canvas not found")?
            .dyn_into::<HtmlCanvasElement>()?;
        canvas.set_width(width);
        canvas.set_height(height);

        let ctx2d = canvas
            .get_context("2d")?
            .unwrap()
            .dyn_into::<CanvasRenderingContext2d>()?;

        Ok(Self { canvas, ctx2d, width, height })
    }

    /// Blit packed RGBA u32 pixels onto the canvas.
    pub fn draw(&self, pixels: &[u32]) -> Result<(), JsValue> {
        // Unpack u32 → RGBA bytes
        let mut bytes = Vec::with_capacity(pixels.len() * 4);
        for &p in pixels {
            bytes.push((p & 0xFF) as u8);
            bytes.push(((p >> 8) & 0xFF) as u8);
            bytes.push(((p >> 16) & 0xFF) as u8);
            bytes.push(((p >> 24) & 0xFF) as u8);
        }

        let clamped = wasm_bindgen::Clamped(&bytes[..]);
        let image_data = ImageData::new_with_u8_clamped_array_and_sh(
            clamped,
            self.width,
            self.height,
        )?;
        self.ctx2d.put_image_data(&image_data, 0.0, 0.0)?;

        Ok(())
    }
}

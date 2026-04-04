// weights.rs — Load pre-trained model weights from binary files

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, Response};

fn bytes_to_f32(bytes: &[u8]) -> Result<Vec<f32>, JsValue> {
    if bytes.len() % 4 != 0 {
        return Err(JsValue::from_str("Weight file size not multiple of 4"));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

/// Fetch all weight files in parallel using Promise.all().
/// Returns Vec<Vec<f32>> in the same order as `urls`.
pub async fn load_all_parallel(urls: &[String]) -> Result<Vec<Vec<f32>>, JsValue> {
    let window = web_sys::window().unwrap();

    // Build array of fetch promises
    let promises = js_sys::Array::new();
    for url in urls {
        let opts = RequestInit::new();
        opts.set_method("GET");
        let request = Request::new_with_str_and_init(url, &opts)?;
        let promise = window.fetch_with_request(&request);
        promises.push(&promise);
    }

    // Await all fetches in parallel
    let responses = JsFuture::from(js_sys::Promise::all(&promises)).await?;
    let responses = js_sys::Array::from(&responses);

    // Now get all array buffers in parallel
    let buf_promises = js_sys::Array::new();
    for i in 0..responses.length() {
        let resp: Response = responses.get(i).dyn_into()?;
        if !resp.ok() {
            return Err(JsValue::from_str(&format!(
                "Failed to fetch {}: {}", urls[i as usize], resp.status()
            )));
        }
        buf_promises.push(&resp.array_buffer().unwrap());
    }

    let buffers = JsFuture::from(js_sys::Promise::all(&buf_promises)).await?;
    let buffers = js_sys::Array::from(&buffers);

    // Convert all buffers to Vec<f32>
    let mut result = Vec::with_capacity(urls.len());
    for i in 0..buffers.length() {
        let uint8 = js_sys::Uint8Array::new(&buffers.get(i));
        let bytes = uint8.to_vec();
        result.push(bytes_to_f32(&bytes)?);
    }

    Ok(result)
}

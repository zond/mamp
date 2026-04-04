// weights.rs — Load pre-trained model weights from binary files
//
// Export weights from PyTorch:
//   ```python
//   import torch
//   model = torch.load("ha2024_checkpoint.pth")
//   state = model['state_dict'] if 'state_dict' in model else model
//   for name, param in state.items():
//       data = param.cpu().float().numpy()
//       data.tofile(f"weights/{name}.bin")
//       print(f"{name}: {list(data.shape)} -> {data.nbytes} bytes")
//   ```
//
// Then serve the weights/ directory alongside the WASM app and fetch them.

use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, Response};

/// Fetch a binary weight file from the server, return as Vec<f32>.
pub async fn load_weight_file(url: &str) -> Result<Vec<f32>, JsValue> {
    let window = web_sys::window().unwrap();

    let opts = RequestInit::new();
    opts.set_method("GET");

    let request = Request::new_with_str_and_init(url, &opts)?;

    let resp_value: JsValue = JsFuture::from(window.fetch_with_request(&request)).await?;
    let resp: Response = resp_value.dyn_into()?;

    if !resp.ok() {
        return Err(JsValue::from_str(&format!(
            "Failed to fetch {}: {} {}",
            url,
            resp.status(),
            resp.status_text()
        )));
    }

    let buffer = JsFuture::from(resp.array_buffer()?).await?;
    let uint8_array = js_sys::Uint8Array::new(&buffer);
    let bytes = uint8_array.to_vec();

    // Interpret raw bytes as f32
    if bytes.len() % 4 != 0 {
        return Err(JsValue::from_str("Weight file size not multiple of 4"));
    }

    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();

    Ok(floats)
}


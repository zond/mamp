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

    let mut opts = RequestInit::new();
    opts.method("GET");

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

/// Weight manifest: maps layer names to their expected shapes.
/// Use this to validate loaded weights match the architecture.
pub struct WeightManifest {
    pub entries: Vec<(&'static str, Vec<usize>)>,
}

impl WeightManifest {
    /// Manifest for the Ha et al. efficient architecture.
    pub fn ha2024() -> Self {
        Self {
            entries: vec![
                // Encoder
                ("enc_conv1.weight", vec![16, 3, 3, 3]),
                ("enc_conv1.bias", vec![16]),
                ("enc_conv2.weight", vec![32, 16, 3, 3]),
                ("enc_conv2.bias", vec![32]),
                ("enc_conv3.weight", vec![32, 32, 3, 3]),
                ("enc_conv3.bias", vec![32]),
                ("enc_texture.weight", vec![32, 32, 1, 1]),
                ("enc_texture.bias", vec![32]),
                // Decoder
                ("dec_conv1.weight", vec![32, 32, 3, 3]),
                ("dec_conv1.bias", vec![32]),
                ("dec_conv2.weight", vec![16, 32, 3, 3]),
                ("dec_conv2.bias", vec![16]),
                ("dec_conv3.weight", vec![3, 16, 3, 3]),
                ("dec_conv3.bias", vec![3]),
            ],
        }
    }

    /// Validate that a weight vector has the expected number of elements.
    pub fn validate(&self, name: &str, data: &[f32]) -> Result<(), String> {
        for (n, shape) in &self.entries {
            if *n == name {
                let expected: usize = shape.iter().product();
                if data.len() != expected {
                    return Err(format!(
                        "{}: expected {} elements ({:?}), got {}",
                        name, expected, shape, data.len()
                    ));
                }
                return Ok(());
            }
        }
        Err(format!("Unknown weight: {}", name))
    }
}

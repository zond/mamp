// lib.rs — Module declarations only.
// WASM app code is in app.rs (gated behind wasm32).
// Shader tests compile on native targets without web dependencies.

#[cfg(target_arch = "wasm32")]
mod app;
#[cfg(target_arch = "wasm32")]
mod evm;
#[cfg(target_arch = "wasm32")]
mod gpu;
#[cfg(target_arch = "wasm32")]
mod steerable;
#[cfg(target_arch = "wasm32")]
mod video;

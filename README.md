# motion-mag-webgpu

Real-time video motion magnification running entirely in the browser via **Rust → WebAssembly → WebGPU**.

Implements the efficient architecture from [Ha et al. 2024](https://arxiv.org/abs/2403.01898),
which achieves 4.2× fewer FLOPs than the Oh et al. 2018 baseline while maintaining comparable quality.

## Architecture

```
Camera → [RGBA→CHW] → Encoder → Manipulator → Decoder → [CHW→RGBA] → Canvas
  30fps     GPU          GPU         GPU          GPU        GPU        Display
```

**Encoder** (single branch — Ha et al. key finding):
- Conv2D 3→16, k3s1p1, ReLU
- Conv2D 16→32, k3s2p1, ReLU (spatial downsample 2×)
- Conv2D 32→32, k3s1p1, ReLU → shape representation
- Conv2D 32→32, k1s1p0       → texture representation

**Manipulator**:
- `output = texture + α × (shape_b − shape_a)`

**Decoder** (reduced-resolution latent — Ha et al. key finding):
- Conv2D 32→32, k3s1p1, ReLU (at half resolution)
- Bilinear upsample 2×
- Conv2D 32→16, k3s1p1, ReLU
- Conv2D 16→3, k3s1p1 (no activation)

All convolutions run as WebGPU compute shaders dispatched per output pixel.

## Project structure

```
├── Cargo.toml              # Rust deps (wgpu, wasm-bindgen, web-sys)
├── build.sh                # wasm-pack build script
├── index.html              # Browser frontend with controls
├── export_weights.py       # PyTorch → raw f32 weight exporter
└── src/
    ├── lib.rs              # WASM entry, animation loop
    ├── gpu.rs              # WebGPU device, buffer, dispatch helpers
    ├── model.rs            # Ha et al. model (encoder/manipulator/decoder)
    ├── video.rs            # Camera capture + canvas rendering
    ├── weights.rs          # Weight loading from binary files
    └── shaders/
        ├── conv2d.wgsl     # General Conv2D + ReLU compute kernel
        ├── manipulator.wgsl # Motion diff + amplification
        ├── upsample.wgsl   # Bilinear 2× upsampling
        └── frame_io.wgsl   # RGBA ↔ CHW tensor conversion
```

## Build

```bash
# Prerequisites
cargo install wasm-pack
rustup target add wasm32-unknown-unknown

# Build
chmod +x build.sh
./build.sh

# Serve (needs HTTPS for camera + WebGPU)
cd pkg
npx http-server -S -C cert.pem -K key.pem -p 8443
```

## Loading trained weights

The model ships with random placeholder weights. To use real weights:

1. Train or obtain a checkpoint from the [Ha et al. codebase](https://arxiv.org/abs/2403.01898)
2. Export: `python export_weights.py --checkpoint model.pth --outdir pkg/weights/`
3. In `lib.rs`, replace `MotionMagModel::new()` with a version that calls
   `weights::load_weight_file()` for each layer

## Production improvements

This skeleton prioritizes clarity over perf. For production:

- **Eliminate CPU readback**: Render the output buffer directly to a WebGPU
  texture, then blit to canvas via a render pipeline. Avoids the GPU→CPU→GPU
  round-trip in the current `process_frame()`.
- **Double-buffer frames**: Overlap GPU inference on frame N with camera capture
  of frame N+1.
- **Fused kernels**: Merge sequential conv layers into fewer dispatches using
  shared memory tiling.
- **Temporal filtering**: Add optional IIR bandpass in a compute shader between
  encoder and manipulator for frequency-selective magnification.
- **Quantize weights**: FP16 storage with FP32 compute cuts memory bandwidth
  in half. wgpu supports `f16` storage buffers on capable GPUs.

## Browser requirements

- Chrome 113+ or Edge 113+ (WebGPU)
- Camera access (HTTPS required)
- Firefox: behind `dom.webgpu.enabled` flag (experimental)

## References

- Ha et al., "Revisiting Learning-based Video Motion Magnification for
  Real-time Processing," 2024. arXiv:2403.01898
- Oh et al., "Learning-based Video Motion Magnification," ECCV 2018.
- Wu et al., "Eulerian Video Magnification for Revealing Subtle Changes
  in the World," SIGGRAPH 2012.

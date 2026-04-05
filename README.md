# MAMP — Motion Amplification

Real-time video motion magnification in the browser using WebGPU.

**[Live Demo →](https://zond.github.io/mamp/)**

Point your camera at something with subtle motion — breathing, a vibrating surface, a candle flame — and the app amplifies the motion so it becomes visible.

## How It Works

**Complex Steerable Pyramid** (Simoncelli & Freeman) with **phase-based motion amplification** (Wadhwa et al. 2013).

Per frame (~92 GPU dispatches):
1. Camera RGBA → YIQ color space (luminance + chrominance)
2. Pad luminance to power-of-2 with mirror padding
3. Forward 2D FFT
4. Decompose into 12 oriented sub-bands (3 scales × 4 orientations)
5. For each sub-band: extract phase → IIR temporal bandpass → amplify → rotate phase
6. Reconstruct via filter accumulation
7. Inverse 2D FFT → modified luminance
8. Recombine with original chrominance → RGBA → render to WebGPU canvas

Phase-based amplification (vs. intensity-based EVM) handles larger motions without ghosting because it amplifies spatial displacement directly rather than pixel intensity changes.

## Controls

- **Amplify**: magnification factor (1-200x)
- **Center Hz / BW Hz**: temporal bandpass frequency band
  - 0.5-3 Hz: breathing, pulse
  - 1-10 Hz: tremor, vibration
  - Higher: structural resonance (limited by camera framerate / Nyquist)
- **Camera**: select front/back camera
- **Res**: processing resolution (lower = faster)
- Click canvas to toggle fullscreen

## Architecture

```
src/
  lib.rs              Module declarations (cfg-gated for wasm32)
  app.rs              WASM entry point, animation loop, JS interop
  gpu.rs              WebGPU device + adapter setup
  steerable.rs        Steerable pyramid pipeline (FFT + phase amplification)
  video.rs            Camera capture via getUserMedia
  shaders/
    fft.wgsl                1D FFT (up to 2048, shared memory)
    steerable_filters.wgsl  Frequency-domain oriented filter bank
    filter_accumulate.wgsl  Apply filter + accumulate (reconstruction)
    phase_amplify.wgsl      Phase extraction + IIR bandpass + amplify
    rgba_to_y.wgsl          RGBA -> YIQ + mirror padding
    yiq_to_rgba.wgsl        Modified Y + I,Q -> RGBA + crop
    blit.wgsl               Fullscreen quad: storage buffer -> canvas
```

## Building

```bash
# Build WASM
wasm-pack build --target web --release --out-dir pkg
cp index.html pkg/
cd pkg && python3 -m http.server 8080
# Open http://localhost:8080

# Run GPU shader tests (requires Vulkan GPU)
cargo test --features native-test -- --nocapture
```

## Tests

33 GPU unit tests verify each shader component:
- FFT: 1D (N up to 2048), 2D (row+col), roundtrip, vs CPU DFT
- Steerable filters: partition of unity, orientation/scale selectivity
- Phase amplification: passthrough, magnitude preservation, temporal filtering
- Color conversion: YIQ roundtrip, mirror padding
- End-to-end: decompose -> reconstruct = original

## Requirements

- Browser: Chrome 113+ or Edge 113+ (WebGPU required)
- Camera: HTTPS or localhost (getUserMedia requirement)
- GPU: WebGPU-capable (most modern GPUs)

## References

- Wadhwa et al. 2013, "Phase-Based Video Motion Processing" (MIT)
- Simoncelli & Freeman, "The Steerable Pyramid" (NYU)
- Wu et al. 2012, "Eulerian Video Magnification" (MIT)

#!/usr/bin/env bash
set -euo pipefail

echo "=== Building motion-mag-webgpu ==="

# Ensure wasm-pack is installed
if ! command -v wasm-pack &>/dev/null; then
    echo "Installing wasm-pack..."
    cargo install wasm-pack
fi

# Build WASM with WebGPU target
wasm-pack build --target web --release --out-dir pkg

# Copy HTML and weights to output
cp index.html pkg/
if [ -d weights ]; then
    cp -r weights pkg/weights
    echo "Copied trained weights to pkg/weights/"
fi

echo ""
echo "=== Build complete ==="
echo "Serve with:  cd pkg && python3 -m http.server 8080"
echo "Open:        https://localhost:8080  (needs HTTPS for camera + WebGPU)"
echo ""
echo "For HTTPS dev server, use:"
echo "  npx http-server pkg -S -C cert.pem -K key.pem -p 8443"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-linux-amd}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
HIP_COMPILER="${CMAKE_HIP_COMPILER:-$ROCM_PATH/bin/amdclang++}"
HIP_ARCHS="${CAPMINER_HIP_ARCHITECTURES:-gfx1030;gfx1031;gfx1032;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201}"

command -v cmake >/dev/null || { echo "cmake is required" >&2; exit 1; }
[[ -x "$HIP_COMPILER" ]] || {
  echo "AMD HIP compiler not found: $HIP_COMPILER" >&2
  echo "Install ROCm/HIP, or set CMAKE_HIP_COMPILER and ROCM_PATH." >&2
  exit 1
}

GENERATOR=()
if command -v ninja >/dev/null; then
  GENERATOR=(-G Ninja)
fi

printf 'Building CapMiner AMD/HIP for Linux\n'
printf '  HIP compiler: %s\n' "$HIP_COMPILER"
printf '  AMD targets: %s\n' "$HIP_ARCHS"
printf '  Build directory: %s\n' "$BUILD_DIR"

cmake -S "$ROOT" -B "$BUILD_DIR" "${GENERATOR[@]}" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCMAKE_PREFIX_PATH="$ROCM_PATH" \
  -DCMAKE_HIP_COMPILER="$HIP_COMPILER" \
  -DCAPMINER_ENABLE_CUDA=OFF \
  -DCAPMINER_ENABLE_HIP=ON \
  -DCAPMINER_ENABLE_OPENCL=OFF \
  -DCAPMINER_HIP_ARCHITECTURES="$HIP_ARCHS"

cmake --build "$BUILD_DIR" --parallel "$JOBS"

BIN="$BUILD_DIR/capminer"
[[ -x "$BIN" ]] || { echo "Build finished but $BIN was not created" >&2; exit 1; }
strip "$BIN" 2>/dev/null || true
sha256sum "$BIN" | tee "$BIN.sha256"
echo "AMD Linux miner: $BIN"

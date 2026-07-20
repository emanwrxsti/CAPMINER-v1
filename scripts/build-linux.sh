#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-linux}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"

command -v cmake >/dev/null || { echo "cmake is required" >&2; exit 1; }
command -v nvcc >/dev/null || { echo "nvcc is required. Install the NVIDIA CUDA Toolkit." >&2; exit 1; }

NVCC_RELEASE="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | tail -1)"
NVCC_MAJOR="${NVCC_RELEASE%%.*}"
NVCC_MINOR="${NVCC_RELEASE#*.}"
if [[ -z "$NVCC_RELEASE" || "$NVCC_MAJOR" == "$NVCC_RELEASE" ]]; then
  echo "Unable to determine the CUDA Toolkit version from nvcc --version" >&2
  exit 1
fi

# CUDA 13 removed offline compilation for Maxwell/Pascal/Volta. CUDA 12.8
# can create one broader fat binary that also includes Blackwell sm_120.
if [[ -z "${CAPMINER_CUDA_ARCHITECTURES:-}" ]]; then
  if (( NVCC_MAJOR >= 13 )); then
    CAPMINER_CUDA_ARCHITECTURES="75;86;89;120"
  else
    CAPMINER_CUDA_ARCHITECTURES="52;61;75;86;89;120"
  fi
fi

LEGACY_ARCH_RE='(^|;)(50|52|60|61|62|70|72)(;|$)'
if (( NVCC_MAJOR >= 13 )) && [[ "$CAPMINER_CUDA_ARCHITECTURES" =~ $LEGACY_ARCH_RE ]]; then
  echo "CUDA $NVCC_RELEASE cannot compile Maxwell, Pascal, or Volta targets." >&2
  echo "Use CUDA 12.8/12.x or remove the legacy architectures." >&2
  exit 1
fi
if [[ ";$CAPMINER_CUDA_ARCHITECTURES;" == *";120;"* ]] &&    (( NVCC_MAJOR < 12 || (NVCC_MAJOR == 12 && NVCC_MINOR < 8) )); then
  echo "sm_120 requires CUDA 12.8 or newer." >&2
  exit 1
fi

GENERATOR=()
if command -v ninja >/dev/null; then
  GENERATOR=(-G Ninja)
fi

printf 'Building CapMiner for Linux\n'
printf '  CUDA toolkit: %s\n' "$(nvcc --version | tail -1)"
printf '  Architectures: %s\n' "$CAPMINER_CUDA_ARCHITECTURES"
printf '  Build directory: %s\n' "$BUILD_DIR"

cmake -S "$ROOT" -B "$BUILD_DIR" "${GENERATOR[@]}" \
  -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
  -DCAPMINER_ENABLE_CUDA=ON \
  -DCAPMINER_ENABLE_OPENCL="${CAPMINER_ENABLE_OPENCL:-OFF}" \
  -DCAPMINER_CUDA_ARCHITECTURES="$CAPMINER_CUDA_ARCHITECTURES"

cmake --build "$BUILD_DIR" --parallel "$JOBS"

BIN="$BUILD_DIR/capminer"
[[ -x "$BIN" ]] || { echo "Build finished but $BIN was not created" >&2; exit 1; }
strip "$BIN" 2>/dev/null || true
sha256sum "$BIN" | tee "$BIN.sha256"
echo "Linux miner: $BIN"

#!/usr/bin/env bash
set -euo pipefail
# CUDA 13-compatible build: Turing and newer.
export CAPMINER_CUDA_ARCHITECTURES="${CAPMINER_CUDA_ARCHITECTURES:-75;86;89;120}"
exec "$(dirname "$0")/build-linux.sh" "$@"

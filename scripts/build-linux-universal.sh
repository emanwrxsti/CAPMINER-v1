#!/usr/bin/env bash
set -euo pipefail
# Requires CUDA 12.8/12.x. Produces one NVIDIA Linux binary for:
# Maxwell GTX 900 (sm_52), Pascal GTX 10 (sm_61), Turing RTX 20/GTX 16
# (sm_75), Ampere RTX 30 (sm_86), Ada RTX 40 (sm_89), Blackwell RTX 50
# (sm_120).
export CAPMINER_CUDA_ARCHITECTURES="${CAPMINER_CUDA_ARCHITECTURES:-52;61;75;86;89;120}"
exec "$(dirname "$0")/build-linux.sh" "$@"

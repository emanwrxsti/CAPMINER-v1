#!/usr/bin/env bash
set -euo pipefail
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build python3 pkg-config
cat <<'MSG'
Host build dependencies installed.
Install the NVIDIA CUDA Toolkit separately, then run:
  ./scripts/build-linux.sh
MSG

#!/usr/bin/env bash
# Build and run the full host-side verification for the Alphanumeric miner.
# No GPU or CUDA toolkit required. Exit code 0 = everything passed.
#
#   1. host_kat_test      exact device math (host compilation path) vs the
#                         independent byte reference + loop-control models
#   2. cross_check        official blake3 python library vs both paths
#                         (skipped politely if pip blake3 is missing)
#   3. shim type check    g++ -fsyntax-only over the real .cu with a CUDA
#                         API shim, so typos can't survive to the nvcc build
set -euo pipefail
cd "$(dirname "$0")/.."

CXX="${CXX:-g++}"

if printf 'int main() { return 0; }\n' | "$CXX" -x c++ -std=c++20 -fsyntax-only - >/dev/null 2>&1; then
    CXX_STD_FLAG="-std=c++20"
elif printf 'int main() { return 0; }\n' | "$CXX" -x c++ -std=c++2a -fsyntax-only - >/dev/null 2>&1; then
    CXX_STD_FLAG="-std=c++2a"
else
    echo "ERROR: $CXX does not support C++20." >&2
    echo "Install GCC 10+ or Clang 10+." >&2
    exit 1
fi

echo "Using $CXX $CXX_STD_FLAG"

mkdir -p build_tests

echo "== [1/3] building and running host_kat_test =="
"$CXX" -O2 "$CXX_STD_FLAG" -Wall -Wextra -I src -o build_tests/host_kat_test tests/host_kat_test.cpp
./build_tests/host_kat_test

echo
echo "== [2/3] blake3 library cross-check =="
python3 tests/cross_check_blake3.py

echo
echo "== [3/3] backend .cu syntax/type check (CUDA shim) =="
sed 's/\([A-Za-z_][A-Za-z0-9_]*\)<<<\([^>]*\)>>>/(ALPHA_SHIM_CFG(\2), \1)/g' \
    src/alphanumeric/alphanumeric_cuda_backend.cu > build_tests/backend_shim.cpp
"$CXX" -fsyntax-only "$CXX_STD_FLAG" -Wall -Wextra -Wno-unknown-pragmas \
    -I tests/cuda_syntax_check -I src -I src/alphanumeric build_tests/backend_shim.cpp
echo "backend .cu type check OK"

echo
echo "ALL HOST-SIDE CHECKS PASSED"

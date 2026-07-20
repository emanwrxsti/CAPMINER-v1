@echo off
REM Alphanumeric RTX 5080 CUDA BLAKE3-92 benchmark/self-test.
REM This does not connect to a pool or submit blocks.
capminer.exe --algo alphanumeric --benchmark --devices 0 --threads 256 --blocks-per-sm 24 --bench-seconds 5 --no-opencl
pause

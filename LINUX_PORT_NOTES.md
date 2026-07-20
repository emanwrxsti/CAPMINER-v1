# Linux port notes

The Linux port changes only platform integration and build packaging. The
Alphanumeric CUDA hash math and share-submission logic are not rewritten.

## Platform changes

- Removed the Windows-only CMake guard.
- Made CUDA optional at configure time so host tests can compile without nvcc.
- Added configurable CUDA fat-binary architecture targets.
- Added POSIX TCP sockets with partial-send handling, EINTR retry, IPv4/IPv6
  resolution, and SIGPIPE-safe sends.
- Added Linux NVML telemetry loading through `dlopen("libnvidia-ml.so.1")`.
- Linked Linux builds with pthreads and the platform dynamic-loader library.
- Added native Linux start, build, package, and GitHub Actions scripts.
- Added a no-CUDA stub build for compiler/CI validation.

## Validation performed in this package

- GCC 14/CMake/Ninja Linux host build completed successfully.
- Host Alphanumeric known-answer, serialization, word-path, head/tail,
  target-compare, loop-accounting, and slice-planner tests passed.
- The Alphanumeric CUDA source passed the repository's CUDA API shim syntax
  check.

A real CUDA Linux binary still needs nvcc. The included GitHub Actions workflow
uses official NVIDIA CUDA development containers to compile downloadable Linux
artifacts without requiring a GPU on the runner.

# Included Windows binary

`capminer.exe` is the original Windows release found in the uploaded source
archive. It was compiled for RTX 50 / `sm_120` and is kept here for reference.

To produce Windows binaries for other NVIDIA generations, build the source with:

- `BUILD_WINDOWS_MODERN.bat` for RTX 20/30/40/50 and GTX 16
- `BUILD_WINDOWS_UNIVERSAL.bat` with CUDA 12.8 for most GTX 900/10/16 and RTX 20/30/40/50

The GitHub Linux workflow builds separate native Linux artifacts.

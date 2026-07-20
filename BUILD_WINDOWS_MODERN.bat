@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo Building modern NVIDIA Windows release...
echo Targets: RTX 20/GTX 16, RTX 30, RTX 40, RTX 50
where cmake >nul 2>&1 || (echo ERROR: cmake not found & exit /b 1)
where nvcc >nul 2>&1 || (echo ERROR: nvcc not found & exit /b 1)

cmake -S . -B build-modern -G "Visual Studio 17 2022" -A x64 ^
  -DCAPMINER_ENABLE_CUDA=ON ^
  -DCAPMINER_ENABLE_OPENCL=OFF ^
  -DCAPMINER_CUDA_ARCHITECTURES="75;86;89;120"
if errorlevel 1 exit /b 1
cmake --build build-modern --config Release --parallel
if errorlevel 1 exit /b 1

echo Built: %CD%\build-modern\Release\capminer.exe

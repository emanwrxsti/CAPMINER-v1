@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo Building universal NVIDIA Windows release with CUDA 12.x...
echo Targets: GTX 900/10/16 and RTX 20/30/40/50
where cmake >nul 2>&1 || (echo ERROR: cmake not found & exit /b 1)
where nvcc >nul 2>&1 || (echo ERROR: nvcc not found & exit /b 1)

for /f "tokens=5" %%V in ('nvcc --version ^| findstr /C:"release"') do set CUDA_REL=%%V

echo CUDA: %CUDA_REL%
echo NOTE: CUDA 13 cannot target Maxwell/Pascal. Use CUDA 12.8 for this build.

cmake -S . -B build-universal -G "Visual Studio 17 2022" -A x64 ^
  -DCAPMINER_ENABLE_CUDA=ON ^
  -DCAPMINER_ENABLE_OPENCL=OFF ^
  -DCAPMINER_CUDA_ARCHITECTURES="52;61;75;86;89;120"
if errorlevel 1 exit /b 1
cmake --build build-universal --config Release --parallel
if errorlevel 1 exit /b 1

echo Built: %CD%\build-universal\Release\capminer.exe

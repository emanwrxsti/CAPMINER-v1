@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo ============================================================
echo Building ICMINERS CAPMINER - Fixed Release
echo ============================================================

where cmake >nul 2>&1
if errorlevel 1 (
    echo ERROR: CMake was not found in PATH.
    echo Open Developer PowerShell for Visual Studio 2022 and run this file again.
    pause
    exit /b 1
)

cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCAPMINER_ENABLE_CUDA=ON -DCAPMINER_ENABLE_OPENCL=OFF
if errorlevel 1 goto :failed

cmake --build build --config Release
if errorlevel 1 goto :failed

echo.
echo Build completed:
echo   %CD%\build\Release\capminer.exe
echo.
pause
exit /b 0

:failed
echo.
echo BUILD FAILED. Review the error above.
pause
exit /b 1

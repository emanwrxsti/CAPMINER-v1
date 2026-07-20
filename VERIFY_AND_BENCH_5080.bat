@echo off
REM ============================================================================
REM VERIFY_AND_BENCH_5080.bat
REM One command on the RTX 5080 box: clean Release build, sm_120 arch check,
REM on-device correctness suite, baseline benchmark, batch-size sweep.
REM Run from "Developer PowerShell/Command Prompt for VS 2022" in capminer\.
REM Exit code 0 = build OK + verify PASS + benchmarks completed.
REM ============================================================================
setlocal
cd /d "%~dp0"

echo == [1/5] clean CMake Release build (CUDA sm_120) ==
if exist build\CMakeCache.txt del /q build\CMakeCache.txt
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCAPMINER_ENABLE_CUDA=ON -DCAPMINER_ENABLE_OPENCL=OFF
if errorlevel 1 exit /b 1
cmake --build build --config Release
if errorlevel 1 exit /b 1

echo.
echo == [2/5] embedded CUDA arch + kernel resource usage ==
cuobjdump --list-elf build\Release\capminer.exe | findstr sm_
cuobjdump --dump-resource-usage build\Release\capminer.exe | findstr /C:"alpha_scan" /C:"REG" /C:"SMEM" /C:"LMEM"

echo.
echo == [3/5] on-device correctness: --verify 2000 ==
build\Release\capminer.exe --algo alphanumeric --verify 2000 --devices 0 --threads 512 --blocks-per-sm 4 --no-opencl
if errorlevel 1 (
    echo VERIFY FAILED - do not mine with this build.
    exit /b 4
)

echo.
echo == [4/5] baseline benchmark (batch 2^24, comparable to the 12.302 GH/s run) ==
build\Release\capminer.exe --algo alphanumeric --benchmark --bench-batch-log2 24 --devices 0 --threads 512 --blocks-per-sm 4 --bench-seconds 10 --no-opencl
if errorlevel 1 exit /b 1

echo.
echo == [5/5] batch-size sweep 2^24..2^28 ==
build\Release\capminer.exe --algo alphanumeric --bench-sweep --devices 0 --threads 512 --blocks-per-sm 4 --bench-seconds 5 --no-opencl
if errorlevel 1 exit /b 1

echo.
echo ALL GPU CHECKS PASSED. To refresh the release binary:
echo   copy /Y build\Release\capminer.exe release-package\capminer.exe
endlocal

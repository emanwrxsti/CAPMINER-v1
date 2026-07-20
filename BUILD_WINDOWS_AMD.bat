@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"

rem AMD HIP SDK for Windows does not support CMake's HIP language.
rem This script invokes hipcc directly and builds an Alphanumeric AMD binary.

if not defined HIP_PATH (
  for /f "delims=" %%D in ('dir /b /ad /o-n "C:\Program Files\AMD\ROCm" 2^>nul') do (
    if not defined HIP_PATH set "HIP_PATH=C:\Program Files\AMD\ROCm\%%D"
  )
)

if not defined HIP_PATH (
  echo ERROR: HIP_PATH is not set and no AMD HIP SDK was found.
  echo Install the AMD HIP SDK, then set HIP_PATH to its install folder.
  exit /b 1
)

set "HIPCC=%HIP_PATH%\bin\hipcc.bat"
if not exist "%HIPCC%" set "HIPCC=%HIP_PATH%\bin\hipcc.exe"
if not exist "%HIPCC%" (
  echo ERROR: hipcc was not found under "%HIP_PATH%\bin".
  exit /b 1
)

set "CLANGC=%HIP_PATH%\bin\clang.exe"
if not exist "%CLANGC%" (
  echo ERROR: clang.exe was not found under "%HIP_PATH%\bin".
  exit /b 1
)

if not defined HIP_ARCH_FLAGS set "HIP_ARCH_FLAGS=--offload-arch=gfx1030 --offload-arch=gfx1031 --offload-arch=gfx1032 --offload-arch=gfx1100 --offload-arch=gfx1101 --offload-arch=gfx1102 --offload-arch=gfx1200 --offload-arch=gfx1201"

if not exist build-amd mkdir build-amd
set "RSP=build-amd\capminer-amd.rsp"
set "WHIRLPOOL_OBJ=build-amd\whirlpool_c.obj"

echo Compiling the legacy Whirlpool implementation as C...
"%CLANGC%" -std=c99 -O3 -DNDEBUG -Isrc -c src/crypto/whirlpool/whirlpool.c -o "%WHIRLPOOL_OBJ%"
if errorlevel 1 exit /b 1

>"%RSP%" echo -std=c++20 -O3 -DNDEBUG
>>"%RSP%" echo -DWIN32_LEAN_AND_MEAN -DNOMINMAX -DCAPMINER_HIP=1 -DCAPMINER_GPU_BACKEND_HIP=1
>>"%RSP%" echo -Isrc
>>"%RSP%" echo %HIP_ARCH_FLAGS%
>>"%RSP%" echo src/main.cpp
>>"%RSP%" echo src/config.cpp
>>"%RSP%" echo src/logger.cpp
>>"%RSP%" echo src/stratum_client.cpp
>>"%RSP%" echo src/stats.cpp
>>"%RSP%" echo src/share_validator.cpp
>>"%RSP%" echo src/nvml_telemetry.cpp
>>"%RSP%" echo src/whirlpool/whirlpool.cpp
>>"%RSP%" echo src/whirlpool/capstash_job.cpp
>>"%RSP%" echo src/whirlpool/sha256_simple.cpp
>>"%RSP%" echo src/whirlpool/capstash_pow.cpp
>>"%RSP%" echo src/alphanumeric/alphanumeric_runner.cpp
>>"%RSP%" echo src/crypto/whirlpool.cpp
>>"%RSP%" echo %WHIRLPOOL_OBJ%
>>"%RSP%" echo src/cuda_backend_stub.cpp
>>"%RSP%" echo src/alphanumeric/alphanumeric_hip_backend.hip
>>"%RSP%" echo -lws2_32
>>"%RSP%" echo -o build-amd/capminer-amd.exe

echo Building AMD HIP Windows release...
echo HIP SDK: %HIP_PATH%
echo Targets: %HIP_ARCH_FLAGS%
call "%HIPCC%" @"%RSP%"
if errorlevel 1 exit /b 1

if not exist build-amd\capminer-amd.exe (
  echo ERROR: build finished without build-amd\capminer-amd.exe
  exit /b 1
)

echo Built: %CD%\build-amd\capminer-amd.exe
endlocal

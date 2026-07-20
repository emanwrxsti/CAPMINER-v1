@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title ICMINERS CAPMINER - Alpha 315W

rem ============================================================
rem EDIT THESE SETTINGS IF NEEDED
rem ============================================================
set "POOL=stratum+tcp://us.icminers.com:7182"
set "WALLET=YOUR_ALPHA_WALLET"
set "WORKER=5080"
set "PASSWORD=x"
set "DEVICE=0"
set "SUBMIT_FORMAT=compact-decstr"

rem RTX 5080 power and CUDA tuning
set "POWER_LIMIT_W=315"
set "THREADS=256"
set "BLOCKS_PER_SM=24"
set "BATCH_MS=100"
set "LOG_FILE=alpha_debug.log"

rem ============================================================
rem SET NVIDIA POWER LIMIT
rem This raises the cap; it does not force the GPU to consume 315W.
rem Run this BAT as Administrator or the driver can reject the change.
rem ============================================================
set "NVIDIA_SMI=%SystemRoot%\System32\nvidia-smi.exe"
if not exist "%NVIDIA_SMI%" set "NVIDIA_SMI=nvidia-smi"

echo Setting GPU %DEVICE% power limit to %POWER_LIMIT_W% W...
"%NVIDIA_SMI%" -i %DEVICE% -pl %POWER_LIMIT_W%
if errorlevel 1 (
    echo WARNING: Could not set the power limit.
    echo Right-click this BAT and choose Run as administrator.
    echo Mining will continue with the current driver power limit.
) else (
    "%NVIDIA_SMI%" -i %DEVICE% --query-gpu=power.limit --format=csv,noheader
)

echo.

rem ============================================================
rem FIND CAPMINER.EXE AUTOMATICALLY
rem ============================================================
set "CAPMINER_EXE="
if exist "%~dp0capminer.exe" set "CAPMINER_EXE=%~dp0capminer.exe"
if not defined CAPMINER_EXE if exist "%~dp0Release\capminer.exe" set "CAPMINER_EXE=%~dp0Release\capminer.exe"
if not defined CAPMINER_EXE if exist "%~dp0build\Release\capminer.exe" set "CAPMINER_EXE=%~dp0build\Release\capminer.exe"

if not defined CAPMINER_EXE (
    echo ERROR: capminer.exe was not found.
    echo Put this BAT beside capminer.exe or in the capminer source folder.
    pause
    exit /b 1
)

echo ============================================================
echo ICMINERS CAPMINER
echo ============================================================
echo Executable: %CAPMINER_EXE%
echo Pool:       %POOL%
echo Wallet:     %WALLET%
echo Worker:     %WORKER%
echo Device:     %DEVICE%
echo Power cap:  %POWER_LIMIT_W% W
echo Threads:    %THREADS%
echo Blocks/SM:  %BLOCKS_PER_SM%
echo Batch:      %BATCH_MS% ms
echo ============================================================
echo.

"%CAPMINER_EXE%" ^
 --algo alphanumeric ^
 --pool "%POOL%" ^
 --wallet "%WALLET%" ^
 --worker "%WORKER%" ^
 --pass "%PASSWORD%" ^
 --devices %DEVICE% ^
 --threads %THREADS% ^
 --blocks-per-sm %BLOCKS_PER_SM% ^
 --batch-ms %BATCH_MS% ^
 --alpha-submit-format %SUBMIT_FORMAT% ^
 --log-file "%LOG_FILE%" ^
 --no-opencl

set "EXIT_CODE=%ERRORLEVEL%"
echo.
echo Capminer stopped with exit code %EXIT_CODE%.
echo Review %LOG_FILE% if it stopped unexpectedly.
pause
exit /b %EXIT_CODE%

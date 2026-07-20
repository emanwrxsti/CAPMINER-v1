@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title ICMINERS CAPMINER - Alphanumeric

rem ============================================================
rem EDIT THESE SETTINGS IF NEEDED
rem ============================================================
set "POOL=stratum+tcp://us.icminers.com:7182"
set "WALLET=YOUR_ALPHA_WALLET"
set "WORKER=5080"
set "PASSWORD=x"
set "DEVICE=0"

rem Keep compact-decstr for the current Miningcore Alpha setup.
set "SUBMIT_FORMAT=compact-decstr"

rem RTX 5080 tuning defaults
set "THREADS=256"
set "BLOCKS_PER_SM=24"
set "BATCH_MS=100"
set "LOG_FILE=alpha_debug.log"

rem ============================================================
rem FIND CAPMINER.EXE AUTOMATICALLY
rem ============================================================
set "CAPMINER_EXE="

if exist "%~dp0capminer.exe" set "CAPMINER_EXE=%~dp0capminer.exe"
if not defined CAPMINER_EXE if exist "%~dp0Release\capminer.exe" set "CAPMINER_EXE=%~dp0Release\capminer.exe"
if not defined CAPMINER_EXE if exist "%~dp0build\Release\capminer.exe" set "CAPMINER_EXE=%~dp0build\Release\capminer.exe"

if not defined CAPMINER_EXE (
    echo.
    echo ERROR: capminer.exe was not found.
    echo.
    echo Put START_MINING.bat beside capminer.exe, or keep the compiled
    echo executable in Release\capminer.exe or build\Release\capminer.exe.
    echo.
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
echo Submit:     %SUBMIT_FORMAT%
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

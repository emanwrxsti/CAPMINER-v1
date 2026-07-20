@echo off
setlocal EnableExtensions
cd /d "%~dp0"

set "POOL=stratum+tcp://us.icminers.com:7182"
set "WALLET=YOUR_ALPHA_WALLET"
set "WORKER=amd-windows-rig"
set "DEVICE=0"
set "THREADS=256"
set "BLOCKS_PER_CU=8"
set "BATCH_MS=100"

if "%WALLET%"=="YOUR_ALPHA_WALLET" (
  echo Edit START_MINING_AMD.bat and set WALLET first.
  pause
  exit /b 1
)

build-amd\capminer-amd.exe ^
 --algo alphanumeric ^
 --pool "%POOL%" ^
 --wallet "%WALLET%" ^
 --worker "%WORKER%" ^
 --pass x ^
 --devices "%DEVICE%" ^
 --threads "%THREADS%" ^
 --blocks-per-sm "%BLOCKS_PER_CU%" ^
 --batch-ms "%BATCH_MS%" ^
 --alpha-submit-format compact-decstr ^
 --no-opencl ^
 --log-file alpha-amd-windows.log

pause
endlocal

@echo off
REM Build and run the host-side verification on Windows (MSVC, no GPU needed).
REM Run from a "x64 Native Tools Command Prompt for VS 2022".
REM Exit code 0 = everything passed.
setlocal
cd /d "%~dp0.."

if not exist build_tests mkdir build_tests

echo == [1/2] building and running host_kat_test ==
cl /nologo /O2 /std:c++20 /EHsc /W3 /I src /Fe:build_tests\host_kat_test.exe /Fo:build_tests\ tests\host_kat_test.cpp
if errorlevel 1 exit /b 1
build_tests\host_kat_test.exe
if errorlevel 1 exit /b 1

echo.
echo == [2/2] blake3 library cross-check (needs: pip install blake3) ==
python tests\cross_check_blake3.py build_tests\host_kat_test.exe
if errorlevel 1 exit /b 1

echo.
echo ALL HOST-SIDE CHECKS PASSED
endlocal

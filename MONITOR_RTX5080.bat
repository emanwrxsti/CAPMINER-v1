@echo off
title RTX 5080 Mining Monitor
"%SystemRoot%\System32\nvidia-smi.exe" --query-gpu=timestamp,name,utilization.gpu,clocks.sm,clocks.mem,power.draw,power.limit,temperature.gpu,fan.speed --format=csv -l 2
pause

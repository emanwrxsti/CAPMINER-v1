@echo off
capminer.exe --pool stratum+tcp://eu.icminers.com:PORT --wallet WALLET_ADDRESS --worker rig1 --pass x --devices 0 --intensity 20 --no-opencl --log-file capminer-eu.log
pause

# CapMiner native Linux builds

CapMiner runs natively on Linux x86-64; Wine is not required.

## NVIDIA CUDA

Modern build for RTX 20/30/40/50 and GTX 16:

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build python3
chmod +x scripts/*.sh START_MINING*.sh tests/build_and_run.sh
./tests/build_and_run.sh
./scripts/build-linux-modern.sh
```

Universal CUDA 12.x build adding GTX 900 and GTX 10:

```bash
./scripts/build-linux-universal.sh
```

Outputs are under `build-linux-modern/` or `build-linux-universal/`.

## AMD HIP/ROCm

Install a compatible AMD driver and ROCm/HIP toolchain, then:

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build python3
chmod +x scripts/*.sh START_MINING*.sh tests/build_and_run.sh
./tests/build_and_run.sh
./scripts/build-linux-amd.sh
```

The default AMD build includes RDNA2, RDNA3, and RDNA4 code objects. Override
the target list through `CAPMINER_HIP_ARCHITECTURES` when necessary.

Output:

```text
build-linux-amd/capminer
build-linux-amd/capminer.sha256
```

Run:

```bash
cp build-linux-amd/capminer .
WALLET=YOUR_ALPHA_WALLET DEVICES=0 WORKER=amd0 ./START_MINING_AMD.sh
```

## Multi-GPU

Run one process per GPU:

```bash
WALLET=YOUR_ALPHA_WALLET DEVICES=0 WORKER=gpu0 ./START_MINING.sh &
WALLET=YOUR_ALPHA_WALLET DEVICES=1 WORKER=gpu1 ./START_MINING.sh &
wait
```

Use `START_MINING_AMD.sh` for AMD binaries. One process per GPU allows separate
thread, block, and batch tuning for mixed models.

See `GPU_SUPPORT.md` for the complete target matrix and limitations.

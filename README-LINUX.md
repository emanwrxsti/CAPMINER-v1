# CapMiner Linux build

This repository now supports a native Linux x86-64 build. It does not use Wine.

## Supported NVIDIA build targets

The CUDA architecture list is configurable through `CAPMINER_CUDA_ARCHITECTURES`.

- `sm_120`: RTX 50 / Blackwell
- `sm_89`: RTX 40 / Ada
- `sm_86`: RTX 30 / Ampere
- `sm_75`: RTX 20 and GTX 16 / Turing
- `sm_61`: GTX 10 / Pascal, CUDA 12.x build only
- `sm_52`: most GTX 900 / Maxwell, CUDA 12.x build only

The default script detects the CUDA Toolkit major version. CUDA 13 builds
Turing and newer; CUDA 12.x also includes Maxwell and Pascal targets.

## Ubuntu build

Install a supported NVIDIA driver and CUDA Toolkit, then:

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build python3
chmod +x scripts/*.sh START_MINING.sh tests/build_and_run.sh
./tests/build_and_run.sh
./scripts/build-linux.sh
```

Output:

```text
build-linux/capminer
build-linux/capminer.sha256
```

For a broad CUDA 12.x NVIDIA fat binary:

```bash
./scripts/build-linux-universal.sh
```

For CUDA 13 / Turing-and-newer:

```bash
./scripts/build-linux-modern.sh
```

Override targets manually:

```bash
CAPMINER_CUDA_ARCHITECTURES='86;89;120' ./scripts/build-linux.sh
```

## Run

```bash
cp build-linux/capminer .
chmod +x capminer START_MINING.sh
WALLET=YOUR_WALLET WORKER=rig1 DEVICES=0 ./START_MINING.sh
```

Multiple NVIDIA GPUs should use one miner process per GPU because this
source currently initializes one CUDA device per process:

```bash
WALLET=YOUR_WALLET DEVICES=0 WORKER=gpu0 ./START_MINING.sh &
WALLET=YOUR_WALLET DEVICES=1 WORKER=gpu1 ./START_MINING.sh &
wait
```

This also lets each GPU model use its own tuning values.

## GitHub Actions

Push the repository to GitHub and open the **Actions** tab. The Linux workflow
runs host correctness tests and compiles downloadable Linux NVIDIA artifacts.
No GPU is required to compile them.

## AMD status

The repository can detect an OpenCL runtime, but the Alphanumeric mining engine
is CUDA-only in this source. The existing OpenCL file is not an AMD
Alphanumeric hashing backend. A real AMD release still requires porting and
validating the BLAKE3-92 kernel using HIP or OpenCL; this Linux port does not
claim AMD mining support.

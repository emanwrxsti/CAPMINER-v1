# capminer

Native Windows 10/11 and Linux x86-64 GPU miner source for CapStash and Alphanumeric.

> Linux support was added in this package. See [README-LINUX.md](README-LINUX.md) for build, GPU-target, GitHub Actions, and run instructions.

## What this is

This is a native C++/CUDA miner project. The Alphanumeric CUDA path includes the live, verified BLAKE3-92 engine documented below. The older CapStash/OpenCL scaffold remains in the tree but should be treated separately.

Included:

- CMake + Visual Studio 2022 project
- Native Windows and Linux build support
- no CPU mining
- no dev fee
- no hidden mining
- no auto-start
- no persistence
- Stratum TCP client scaffold
- CapStash `whirlpool` module namespace
- CUDA kernel named `whirlpool_mine`
- OpenCL kernel named `whirlpool_mine`
- Windows examples for ICMiners US/EU and Alphanumeric

## Honest status

The Alphanumeric CUDA engine is the maintained mining path in this source and includes host/GPU verification tests. The CapStash/OpenCL portion is still a scaffold and must not be presented as a production AMD or CapStash miner until its hashing and job path are fully validated.

Do not ship this to miners until:

1. `address_to_scriptpubkey()` is implemented from CapStash Core chainparams.
2. `parse_notify()` is completed for your CapStash Stratum job format.
3. Block header serialization is byte-for-byte identical to CapStash Core.
4. CUDA digest output matches CapStash Core test vectors.
5. Share target and submit payload are verified against your pool.

## Build: Windows / Visual Studio 2022

Install:

- Visual Studio 2022 with Desktop development with C++
- CMake tools for Windows
- NVIDIA CUDA Toolkit
- Optional AMD/OpenCL SDK or vendor runtime

Open "Developer PowerShell for VS 2022":

```powershell
cd capminer
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCAPMINER_ENABLE_CUDA=ON -DCAPMINER_ENABLE_OPENCL=OFF
cmake --build build --config Release
```

Binary:

```text
build\Release\capminer.exe
```

## Build: Linux / CUDA

Install a supported NVIDIA driver, CUDA Toolkit, CMake, Ninja, and a C++ compiler, then:

```bash
chmod +x scripts/*.sh tests/build_and_run.sh START_MINING.sh
./tests/build_and_run.sh
./scripts/build-linux.sh
```

Binary:

```text
build-linux/capminer
```

See [README-LINUX.md](README-LINUX.md) for universal and modern NVIDIA build profiles.

## Run

```bat
capminer.exe --pool stratum+tcp://us.icminers.com:PORT --wallet WALLET_ADDRESS --worker rig1 --pass x --devices 0 --intensity 20 --no-opencl --log-file capminer.log
```

## Required CLI

- `--pool`
- `--wallet`
- `--worker`
- `--pass`
- `--devices`
- `--intensity`
- `--no-cuda`
- `--no-opencl`
- `--log-file`

## Next engineering steps

Use CapStash Core as the source of truth for:

- block header format
- PoW Whirlpool hash
- target/difficulty conversion
- getblocktemplate fields
- share validation/proposal behavior

Recommended workflow:

1. Add a small Windows console test that links or ports CapStash Core hash code.
2. Feed it one known CapStash block header.
3. Confirm hash exactly matches `CapStash-cli getblock HASH`.
4. Port that same function to CUDA.
5. Compare CPU reference test mode only in a unit test, not in the miner loop.
6. Enable `mining.submit` only after hash parity is proven.



## Alphanumeric RTX 5080 CUDA mode

This package includes an **Alphanumeric CUDA BLAKE3-92 engine** and a live Stratum loop:

- `src/alphanumeric/alphanumeric_cuda_backend.cu`
- `src/alphanumeric/alphanumeric_cuda_backend.hpp`
- `src/alphanumeric/alphanumeric_runner.cpp`
- CLI selector: `--algo alphanumeric` or `--coin alphanumeric`

What it does:

- Builds the exact 92-byte Alphanumeric header layout used by the Rust miner:
  - `u32 block_number` little-endian
  - `32 bytes previous_hash`
  - `u64 timestamp` little-endian
  - `u64 nonce` little-endian
  - `u64 difficulty` little-endian
  - `32 bytes merkle_root`
- Hashes that header with the BLAKE3 single-chunk/two-block path. The kernel
  precomputes the six nonce-independent round-0 G mixes per job on the host and
  rejects nearly every nonce from the first output word alone, finishing the
  final round only for real candidates.
- Compares CUDA output against a CPU reference before benchmark/mining work,
  and offers a deeper on-device suite with `--verify N` (random GPU-vs-CPU
  hashes, planted-share scans where hash == target must be found, and exact
  work-accounting checks).
- Benchmarks the RTX 5080 CUDA kernel with `--benchmark` (pick the per-launch
  batch with `--bench-batch-log2 N`) or sweeps batch sizes 2^24..2^28 with
  `--bench-sweep`.
- Reacts to new jobs mid-batch: `mining.notify` / `mining.set_difficulty`
  asynchronously flag the running kernel, which drains within microseconds
  instead of finishing the whole batch.
- Connects to a Stratum/Miningcore endpoint, authorizes, waits for Alphanumeric `mining.notify` jobs, scans nonces on CUDA, and submits found shares with `mining.submit`.

Important: a stock Miningcore coin module will not automatically produce Alphanumeric jobs. Your Miningcore side must emit the custom Alphanumeric notify payload documented in `ALPHANUMERIC_MININGCORE_PROTOCOL.txt` and validate the submit payload. The miner side is now wired; the pool side still has to speak the same job/submit format.

### Build for RTX 5080

Open **Developer PowerShell for VS 2022**:

```powershell
cd capminer
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCAPMINER_ENABLE_CUDA=ON -DCAPMINER_ENABLE_OPENCL=OFF
cmake --build build --config Release
```

CUDA targets are configurable. The Linux build profiles cover `sm_75`, `sm_86`, `sm_89`, and `sm_120`; the CUDA 12.x universal profile also adds `sm_52` and `sm_61`.

### Run Alphanumeric CUDA benchmark

```bat
build\Release\capminer.exe --algo alphanumeric --benchmark --devices 0 --threads 512 --blocks-per-sm 4 --bench-seconds 10 --no-opencl
```

Add `--bench-batch-log2 26` to benchmark a specific nonces-per-launch batch
(default 24, i.e. 2^24), or replace `--benchmark` with `--bench-sweep` to print
a GH/s + latency table for batches 2^24..2^28. The benchmark also verifies the
on-device work accounting on every launch and prints `accounting=OK`.

### Verify GPU correctness on this machine

```bat
build\Release\capminer.exe --algo alphanumeric --verify 2000 --devices 0 --threads 512 --blocks-per-sm 4 --no-opencl
```

Runs 2000 random GPU-vs-CPU hash comparisons (including nonce edge cases 0,
2^32-1, 2^32, 2^64-1), planted-share scans where the target equals the best
hash in the range (the scan must find exactly that nonce, proving hash ==
target counts as a share, across ranges that cross 2^32 and wrap 2^64), and
exact hashes-scanned accounting. Exit code 0 means every check passed.

### Run live Alphanumeric Stratum/Miningcore mode

```bat
build\Release\capminer.exe --algo alphanumeric --pool stratum+tcp://127.0.0.1:3333 --wallet YOUR_WALLET --worker rtx5080 --pass x --devices 0 --threads 512 --blocks-per-sm 4 --batch-ms 15 --alpha-submit-format miningcore --no-opencl
```

Or edit and run:

```bat
examples\alphanumeric_miningcore_5080.bat
```

Submit format options:

- `--alpha-submit-format miningcore`  default, best for a custom Miningcore module
- `--alpha-submit-format extended`
- `--alpha-submit-format compact`
- `--alpha-submit-format bitcoin`

See `ALPHANUMERIC_MININGCORE_PROTOCOL.txt` for exact notify and submit JSON.

## Security/non-malware policy

This project does not include stealth, persistence, auto-start, credential theft, hidden processes, or a developer fee.


### Debugging Alphanumeric Stratum jobs

If the miner connects and shows pool difficulty but hashrate stays at 0 H/s, the pool is not sending a parseable Alphanumeric mining.notify job yet. Run:

```bat
.\Release\capminer.exe --algo alphanumeric --pool stratum+tcp://us.icminers.com:7182 --wallet YOUR_ALPHA_WALLET --worker rtx5080 --pass x --devices 0 --threads 256 --blocks-per-sm 32 --batch-ms 50 --alpha-submit-format miningcore --debug-shares --log-file alpha_debug.log --no-opencl
```

Then inspect `alpha_debug.log`. It prints every JSON line received from the pool as `POOL RX:` and prints the raw `mining.notify` if the job format does not match the Alphanumeric custom format.


### Alphanumeric Miningcore auto submit probe

For the live Alphanumeric/Alpha Miningcore port, start with auto submit probing:

```bat
.\Release\capminer.exe --algo alphanumeric --pool stratum+tcp://us.icminers.com:7182 --wallet YOUR_ALPHA_WALLET --worker rtx5080 --pass x --devices 0 --threads 256 --blocks-per-sm 32 --batch-ms 50 --alpha-submit-format auto --debug-shares --log-file alpha_debug.log --no-opencl
```

When the log prints `ACCEPTED submit format found: <format>`, rerun with that exact `--alpha-submit-format` so the miner stops probing extra submit shapes.

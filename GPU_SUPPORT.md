# GPU and operating-system support

CapMiner builds separate NVIDIA CUDA and AMD HIP/ROCm binaries. CUDA and HIP
must not be enabled in the same build directory.

## NVIDIA

| Family | Examples | CUDA target | Build profile |
|---|---|---:|---|
| Blackwell | RTX 5090/5080/5070/5060 | `sm_120` | modern and universal |
| Ada | RTX 4090/4080/4070/4060 | `sm_89` | modern and universal |
| Ampere | RTX 3090 Ti/3090/3080/3070/3060 | `sm_86` | modern and universal |
| Turing | RTX 2080/2070/2060, GTX 1660/1650 | `sm_75` | modern and universal |
| Pascal | GTX 1080/1070/1060/1050 | `sm_61` | universal, CUDA 12.x only |
| Maxwell | most GTX 980/970/960/950 | `sm_52` | universal, CUDA 12.x only |

CUDA 13 cannot compile Maxwell, Pascal, or Volta device code. Use CUDA 12.x
for the universal NVIDIA build.

### NVIDIA build commands

Windows:

```bat
BUILD_WINDOWS_MODERN.bat
BUILD_WINDOWS_UNIVERSAL.bat
```

Linux:

```bash
./scripts/build-linux-modern.sh
./scripts/build-linux-universal.sh
```

## AMD

AMD Alphanumeric mining uses the HIP/ROCm port in
`src/alphanumeric/alphanumeric_hip_backend.hip`. It implements the same
92-byte BLAKE3 header, target comparison, share discovery, cancellation,
benchmark, and verification interface as the CUDA backend.

Default Radeon code objects:

- RDNA2: `gfx1030`, `gfx1031`, `gfx1032`
- RDNA3: `gfx1100`, `gfx1101`, `gfx1102`
- RDNA4: `gfx1200`, `gfx1201`

Actual runtime support depends on the installed AMD driver/ROCm or HIP SDK.
Older GCN, Polaris, Vega, and RDNA1 cards are not claimed as supported by the
default current-ROCm package. Advanced users can override the target list, but
a compiler accepting a target does not guarantee that the installed runtime
supports that GPU.

### AMD Linux

Install ROCm/HIP and then run:

```bash
./scripts/build-linux-amd.sh
```

Override the targets when needed:

```bash
CAPMINER_HIP_ARCHITECTURES='gfx1030;gfx1100;gfx1201' \
  ./scripts/build-linux-amd.sh
```

Output:

```text
build-linux-amd/capminer
```

### AMD Windows

Install the AMD HIP SDK and use:

```bat
BUILD_WINDOWS_AMD.bat
```

Output:

```text
build-amd\capminer-amd.exe
```

AMD's Windows HIP SDK does not expose CMake's HIP language, so the Windows
script invokes `hipcc` directly.

## Scope

- NVIDIA CUDA supports Alphanumeric and the existing CUDA CapStash backend.
- AMD HIP currently supports the Alphanumeric BLAKE3-92 engine.
- The old OpenCL file only detects OpenCL platforms; it is not used as the AMD
  Alphanumeric miner.
- One process per GPU is recommended so each card can have separate tuning.

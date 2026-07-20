#pragma once
#include <vector>
#include <array>
#include <cstdint>
#include <string>
#include "stats.hpp"

struct CudaResult {
    bool found=false;
    uint32_t nonce=0;
    unsigned char hash[32]{};
};

// List CUDA devices (name + SM version). Returns true if at least one device.
bool cuda_list_devices();

// Human-readable name of a CUDA device (e.g. "NVIDIA GeForce RTX 5080").
// Returns "GPU" if the device can't be queried.
std::string cuda_device_name(int device);

// ---------------------------------------------------------------------------
// Optimized CapStash/Whirlpool mining API.
//
// CapStash PoW does Whirlpool512 over the 80-byte header, then folds 512->256
// by XORing the two halves. The first 64 bytes of the header (block0) are
// constant for a whole job; only the nonce (header bytes 76..79) changes, and
// it lives in the SECOND Whirlpool block. So we:
//   1) precompute the Whirlpool "midstate" (state after compressing block0)
//      ONCE per job, and
//   2) in the hot kernel only run the SECOND Whirlpool compression per nonce.
//
// Usage per job:
//   cuda_setup_job(dev, header80, target);          // once per job
//   cuda_hash_one_nonce(dev, 0, h);                 // optional: verify vs CPU
//   loop: cuda_mine_batch(dev, start, count, r, s); // reuses job constants
// ---------------------------------------------------------------------------

// Upload job constants to the device: precomputes the block0 midstate on the
// GPU (using the same compression as the reference path) and stores the header
// tail + share target in constant memory. Call once whenever the header or
// target changes.
bool cuda_setup_job(
    int device,
    const std::array<uint8_t, 80>& header80,
    const std::array<uint8_t, 32>& target
);

// Run the optimized hot path for a SINGLE nonce against the currently set-up
// job, returning the 32-byte folded PoW hash. Used to validate that the
// optimized GPU path matches the CPU reference (CPU nonce0 == GPU nonce0).
bool cuda_hash_one_nonce(int device, uint32_t nonce, unsigned char out32[32]);

// Scan [start_nonce, start_nonce+count) for a hash <= target (little-endian).
// Requires cuda_setup_job() to have been called for the current job.
// threads / blocks_per_sm override the launch config when > 0 (else the
// compiled defaults are used); threads is clamped to the kernel's launch bound.
bool cuda_mine_batch(
    int device,
    uint32_t start_nonce,
    uint32_t count,
    CudaResult& result,
    MinerStats& stats,
    int threads = 0,
    int blocks_per_sm = 0
);

// Reference (full 80-byte, two-block) Whirlpool PoW hash on the GPU. Kept for
// cross-checking against the CPU and the optimized path. Slow by design; only
// used for one-shot validation, never in the mining loop.
bool cuda_hash_once(int device, const std::array<uint8_t, 80>& header80, unsigned char out32[32]);

// Print kernel resource usage (registers, shared bytes) and occupancy
// (resident blocks/warps per SM) for the given block size. Diagnostic only.
void cuda_print_kernel_info(int device, int threads);

// Throughput benchmark: runs the real mining kernel against a synthetic job
// (target that never matches, so the full nonce range is hashed every launch)
// for ~seconds; returns measured nonces/sec. No Stratum, no submit.
double cuda_benchmark(int device, int threads, int blocks_per_sm, double seconds);

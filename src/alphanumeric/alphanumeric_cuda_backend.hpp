#pragma once
#include <array>
#include <cstdint>
#include <string>
#include "../stats.hpp"

struct AlphanumericCudaResult {
    bool found = false;
    uint64_t nonce = 0;
    unsigned char hash[32]{};
    uint64_t hashes_scanned = 0;
};

// List CUDA devices and return true when at least one device exists.
bool alphanumeric_cuda_list_devices();

// Human-readable CUDA device name.
std::string alphanumeric_cuda_device_name(int device);

// Tear down and recreate the CUDA context after a launch/runtime failure.
bool alphanumeric_cuda_reset(int device);

// CPU reference for the exact Alphanumeric 92-byte BLAKE3 header hash.
// Header serialization:
//   u32 block_number LE
//   [32] previous_hash raw
//   u64 timestamp LE
//   u64 nonce LE
//   u64 difficulty LE
//   [32] merkle_root raw
std::array<uint8_t, 32> alphanumeric_cpu_hash92(
    uint32_t block_number,
    const std::array<uint8_t, 32>& previous_hash,
    uint64_t timestamp,
    uint64_t nonce,
    uint64_t difficulty,
    const std::array<uint8_t, 32>& merkle_root
);

// Hash a single nonce on CUDA. Used for parity checks against the CPU reference.
bool alphanumeric_cuda_hash92(
    int device,
    uint32_t block_number,
    const std::array<uint8_t, 32>& previous_hash,
    uint64_t timestamp,
    uint64_t nonce,
    uint64_t difficulty,
    const std::array<uint8_t, 32>& merkle_root,
    unsigned char out32[32]
);

// Scan [start_nonce, start_nonce + nonce_count) for hash <= target_be.
// target_be is the same fixed-width 32-byte big-endian target used by the
// Alphanumeric Rust miner's lexicographic byte comparison.
bool alphanumeric_cuda_scan92(
    int device,
    uint32_t block_number,
    const std::array<uint8_t, 32>& previous_hash,
    uint64_t timestamp,
    uint64_t difficulty,
    const std::array<uint8_t, 32>& merkle_root,
    const std::array<uint8_t, 32>& target_be,
    uint64_t start_nonce,
    uint64_t nonce_count,
    AlphanumericCudaResult& result,
    MinerStats& stats,
    int threads = 0,
    int blocks_per_sm = 0
);

// Asynchronously tell a running scan batch that the current job is stale.
// Bumps the internal job generation and pushes it to the device on a side
// stream; alpha_scan_kernel notices at its next poll (every 256 hashes per
// thread) and drains within microseconds. Safe to call from any thread at any
// time, including before CUDA init (no-op) and while no scan is running.
// Best effort by design: a lost/late write only means the batch finishes
// normally and the runner's job_seq staleness check discards the result.
void alphanumeric_cuda_request_cancel();

// Result of a fixed-batch throughput benchmark.
struct AlphanumericBenchResult {
    bool ok = false;                // false: CUDA error, see stderr
    bool accounting_ok = false;     // every launch reported exactly `batch` hashes
    uint64_t batch = 0;             // nonces per scan call
    uint64_t launches = 0;          // completed scan calls in the timed window
    uint64_t total_hashes = 0;      // sum of device-reported completed hashes
    double seconds_requested = 0.0;
    double seconds = 0.0;           // measured wall time of the timed window
    double hashes_per_sec = 0.0;    // total_hashes / seconds
    double avg_batch_ms = 0.0;      // wall ms per scan call (kernel + host overhead)
    double launches_per_sec = 0.0;
};

// Synthetic throughput benchmark for the real BLAKE3-92 kernel with a chosen
// per-call batch size. Uses an impossible (all-zero) target so every launch
// scans the full batch; also cross-checks the on-device work accounting.
AlphanumericBenchResult alphanumeric_cuda_benchmark_batch(
    int device, int threads, int blocks_per_sm, double seconds, uint64_t batch_nonces);

// Legacy fixed-batch (2^24) benchmark; returns hashes/sec (0.0 on failure).
double alphanumeric_cuda_benchmark(int device, int threads, int blocks_per_sm, double seconds);

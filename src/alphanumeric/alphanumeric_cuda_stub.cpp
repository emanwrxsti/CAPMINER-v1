#include "alphanumeric_cuda_backend.hpp"
#include "alphanumeric_ref_hash.hpp"
#include <cstring>

bool alphanumeric_cuda_list_devices() { return false; }
std::string alphanumeric_cuda_device_name(int) { return "CUDA disabled"; }
bool alphanumeric_cuda_reset(int) { return false; }

std::array<uint8_t, 32> alphanumeric_cpu_hash92(
    uint32_t block_number,
    const std::array<uint8_t, 32>& previous_hash,
    uint64_t timestamp,
    uint64_t nonce,
    uint64_t difficulty,
    const std::array<uint8_t, 32>& merkle_root) {
    // The byte-path reference is pure host code; it works without CUDA.
    return alpha_ref::hash92_fields(block_number, previous_hash, timestamp,
                                    nonce, difficulty, merkle_root);
}

bool alphanumeric_cuda_hash92(
    int,
    uint32_t,
    const std::array<uint8_t, 32>&,
    uint64_t,
    uint64_t,
    uint64_t,
    const std::array<uint8_t, 32>&,
    unsigned char out32[32]) {
    std::memset(out32, 0, 32);
    return false;
}

bool alphanumeric_cuda_scan92(
    int,
    uint32_t,
    const std::array<uint8_t, 32>&,
    uint64_t,
    uint64_t,
    const std::array<uint8_t, 32>&,
    const std::array<uint8_t, 32>&,
    uint64_t,
    uint64_t,
    AlphanumericCudaResult& result,
    MinerStats&,
    int,
    int) {
    result = AlphanumericCudaResult{};
    return false;
}

void alphanumeric_cuda_request_cancel() {}

AlphanumericBenchResult alphanumeric_cuda_benchmark_batch(int, int, int, double seconds, uint64_t batch_nonces) {
    AlphanumericBenchResult br{};
    br.batch = batch_nonces;
    br.seconds_requested = seconds;
    return br;
}

double alphanumeric_cuda_benchmark(int, int, int, double) { return 0.0; }

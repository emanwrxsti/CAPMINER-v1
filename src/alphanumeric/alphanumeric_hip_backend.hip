#include "alphanumeric_cuda_backend.hpp"
// HIP/ROCm implementation exports the same ABI as the CUDA backend.
// Build CUDA and HIP as separate binaries; do not enable both in one target.
#include "alphanumeric_blake3_core.cuh"
#include "alphanumeric_ref_hash.hpp"
#include <hip/hip_runtime.h>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>

#ifndef CAPMINER_HIP_THREADS
#define CAPMINER_HIP_THREADS 256
#endif
#ifndef CAPMINER_HIP_BLOCKS_PER_CU
#define CAPMINER_HIP_BLOCKS_PER_CU 24
#endif

#define ALPHA_HIP_CK(x) do { \
    hipError_t _e = (x); \
    if(_e != hipSuccess) { \
        std::cerr << "HIP error " << hipGetErrorString(_e) << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        return false; \
    } \
} while(0)

namespace {

using alpha_core::AlphaJobPre;

// -----------------------------------------------------------------------------
// Device <-> host result block.
//
// Layout is locked so the per-launch clear can be a single 16-byte
// hipMemsetAsync over exactly {found, pad0, deficit}. nonce/hash are only
// ever read when found != 0, and found != 0 implies the CAS winner wrote them
// in this launch, so they never need clearing.
//
// deficit counts nonces that were REQUESTED from a thread but not hashed
// because the thread exited early (found flag or job-generation change). The
// host reports completed work as enqueued - deficit, which keeps the
// displayed hashrate exact on the rare early-exit path without any per-nonce
// atomics or a big per-launch reduction on the common full-scan path.
// -----------------------------------------------------------------------------
struct DeviceScanResult {
    unsigned int found;
    unsigned int pad0;
    unsigned long long deficit;
    unsigned long long nonce;
    unsigned char hash[32];
};
static_assert(offsetof(DeviceScanResult, deficit) == 8, "clear window covers found+deficit");
static_assert(offsetof(DeviceScanResult, nonce) == 16, "clear window must stop before nonce");
static_assert(sizeof(DeviceScanResult) == 56, "pinned copy size");
constexpr size_t RESULT_CLEAR_BYTES = 16; // found + pad0 + deficit

struct DevCtx {
    int device = -1;
    bool ready = false;
    int sm_count = 1;
    unsigned char* d_hash = nullptr;
    DeviceScanResult* d_result = nullptr;
    DeviceScanResult* h_result = nullptr;   // pinned
    uint32_t* d_live_gen = nullptr;         // device copy of the job generation
    uint32_t* h_gen_src = nullptr;          // pinned staging for async gen writes
    hipStream_t stream = nullptr;          // mining stream (STAGE 4)
    hipStream_t cancel_stream = nullptr;   // tiny stream for job-cancel writes
};

DevCtx g_ctx;

// Host-side job generation. alphanumeric_cuda_request_cancel() bumps it and
// pushes the new value to the device; a running alpha_scan_kernel notices the
// mismatch at its next poll (every 256 hashes per thread) and drains within
// microseconds, so a big batch no longer delays job switching.
std::atomic<uint32_t> g_job_gen{0};
std::mutex g_cancel_mu;

// A single kernel launch never covers more than 2^31 nonces so the in-kernel
// index arithmetic can be pure 32-bit with no wrap hazard (STAGE 3):
// idx < 2^31 and stride < 2^25, so idx + stride always fits in uint32_t.
// Larger requests are split into sequential slices by the host.
constexpr uint64_t SLICE_MAX = 1ull << 31;

static AlphaJobPre build_job_pre(uint32_t block_number,
                                 const std::array<uint8_t, 32>& previous_hash,
                                 uint64_t timestamp,
                                 uint64_t difficulty,
                                 const std::array<uint8_t, 32>& merkle_root,
                                 const std::array<uint8_t, 32>& target_be) {
    uint32_t w[23];
    uint32_t t[8];
    alpha_core::alpha_header_words_from_fields(block_number, previous_hash.data(),
                                               timestamp, difficulty,
                                               merkle_root.data(), w);
    alpha_core::alpha_target_words_from_be_bytes(target_be.data(), t);
    AlphaJobPre jw{};
    alpha_core::alpha_build_job_pre(w, t, jw);
    return jw;
}

// -----------------------------------------------------------------------------
// Device side
// -----------------------------------------------------------------------------

__device__ __forceinline__ void store32_le_dev(unsigned char* p, uint32_t x) {
    p[0] = (unsigned char)x;
    p[1] = (unsigned char)(x >> 8);
    p[2] = (unsigned char)(x >> 16);
    p[3] = (unsigned char)(x >> 24);
}

__global__ void alpha_hash_one_kernel(AlphaJobPre jw, uint64_t nonce,
                                      unsigned char* __restrict__ out32) {
    if(blockIdx.x != 0 || threadIdx.x != 0) return;
    uint32_t h[8];
    alpha_core::alpha_hash92_words(jw, nonce, h);
    for(int i = 0; i < 8; ++i) store32_le_dev(out32 + i * 4, h[i]);
}

__global__ void __launch_bounds__(512)
alpha_scan_kernel(AlphaJobPre jw,
                  uint64_t start_nonce,
                  uint32_t nonce_count,
                  uint32_t job_gen,
                  DeviceScanResult* __restrict__ result,
                  const uint32_t* __restrict__ live_gen) {
    const uint32_t stride = blockDim.x * gridDim.x;              // STAGE 3: 32-bit
    const uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t t0 = jw.target[0];

    uint32_t poll = 0;      // counts loop-body entries; doubles as the work counter
    bool bailed = false;    // true only when the break skipped hashing this iteration

    for(uint32_t idx = tid; idx < nonce_count; idx += stride) {
        // STAGE 1: pre-increment, so the first global found/gen read happens
        // after 255 hashes per thread instead of on the very first iteration.
        // __ldcg reads through L2, which is coherent with the atomicCAS below
        // and with the host's async generation write.
        if(((++poll & 255u) == 0u) &&
           (result->found != 0u || *live_gen != job_gen)) {
            bailed = true;
            break;
        }

        const uint64_t nonce = start_nonce + (uint64_t)idx;      // u64 wrap is well-defined
        const uint32_t nl = (uint32_t)nonce;
        const uint32_t nh = (uint32_t)(nonce >> 32);

        // STAGE 6: precomputed round-0 partials + first-word fast path.
        uint32_t v[16];
        const uint32_t h0 = alpha_core::alpha_hash92_head(jw, nl, nh, v);
        const uint32_t b0 = alpha_core::alpha_bswap32(h0);
        if(b0 > t0) continue;                                    // ~every nonce leaves here

        // Rare candidate: finish the final round from the retained registers,
        // build the full hash, and run the exact byte-lexicographic compare.
        uint32_t h[8];
        alpha_core::alpha_hash92_tail(jw, v, h);
        const bool le = (b0 < t0) || alpha_core::alpha_words_le_target_tail(h, jw.target);
        if(le) {
            if(atomicCAS(&result->found, 0u, 1u) == 0u) {
                result->nonce = nonce;
#pragma unroll
                for(int i = 0; i < 8; ++i) store32_le_dev(result->hash + i * 4, h[i]);
            }
            break;  // this nonce WAS hashed, so bailed stays false
        }
    }

    // Exact completed-work accounting. HIP wavefront width differs between
    // AMD architectures, so avoid CUDA warp-shuffle assumptions here. The
    // common full-scan path has deficit == 0 and performs no atomic. Only
    // threads that exited early contribute their remaining work.
    const uint32_t expected = (tid < nonce_count)
        ? (nonce_count - 1u - tid) / stride + 1u
        : 0u;
    const uint32_t done = poll - (bailed ? 1u : 0u);
    const uint32_t deficit = expected - done;
    if(deficit != 0u)
        atomicAdd(&result->deficit, (unsigned long long)deficit);
}

// -----------------------------------------------------------------------------
// Context management
// -----------------------------------------------------------------------------

static void destroy_ctx_nothrow() {
    if(!g_ctx.ready && g_ctx.device < 0) { g_ctx = DevCtx{}; return; }
    hipSetDevice(g_ctx.device);
    if(g_ctx.stream) hipStreamSynchronize(g_ctx.stream);
    if(g_ctx.cancel_stream) hipStreamSynchronize(g_ctx.cancel_stream);
    if(g_ctx.d_hash) hipFree(g_ctx.d_hash);
    if(g_ctx.d_result) hipFree(g_ctx.d_result);
    if(g_ctx.d_live_gen) hipFree(g_ctx.d_live_gen);
    if(g_ctx.h_result) hipHostFree(g_ctx.h_result);
    if(g_ctx.h_gen_src) hipHostFree(g_ctx.h_gen_src);
    if(g_ctx.stream) hipStreamDestroy(g_ctx.stream);
    if(g_ctx.cancel_stream) hipStreamDestroy(g_ctx.cancel_stream);
    g_ctx = DevCtx{};
}

bool ensure_ready(int device) {
    if(g_ctx.ready && g_ctx.device == device) {
        ALPHA_HIP_CK(hipSetDevice(device));
        return true;
    }
    if(g_ctx.ready) destroy_ctx_nothrow();

    ALPHA_HIP_CK(hipSetDevice(device));
    g_ctx.device = device;
    g_ctx.sm_count = 1;
    hipDeviceGetAttribute(&g_ctx.sm_count, hipDeviceAttributeMultiprocessorCount, device);
    if(g_ctx.sm_count < 1) g_ctx.sm_count = 1;
    ALPHA_HIP_CK(hipMalloc(&g_ctx.d_hash, 32));
    ALPHA_HIP_CK(hipMalloc(&g_ctx.d_result, sizeof(DeviceScanResult)));
    ALPHA_HIP_CK(hipMalloc(&g_ctx.d_live_gen, sizeof(uint32_t)));
    ALPHA_HIP_CK(hipHostMalloc(&g_ctx.h_result, sizeof(DeviceScanResult)));
    ALPHA_HIP_CK(hipHostMalloc(&g_ctx.h_gen_src, sizeof(uint32_t)));
    // STAGE 4: one nonblocking stream owns the scan pipeline; a second tiny
    // stream carries job-cancel generation writes so they can overtake a
    // running kernel instead of queueing behind it.
    ALPHA_HIP_CK(hipStreamCreateWithFlags(&g_ctx.stream, hipStreamNonBlocking));
    ALPHA_HIP_CK(hipStreamCreateWithFlags(&g_ctx.cancel_stream, hipStreamNonBlocking));
    const uint32_t gen_now = g_job_gen.load(std::memory_order_acquire);
    ALPHA_HIP_CK(hipMemcpy(g_ctx.d_live_gen, &gen_now, sizeof(uint32_t), hipMemcpyHostToDevice));
    g_ctx.ready = true;
    return true;
}

static int clamp_threads(int t) {
    if(t <= 0) t = CAPMINER_HIP_THREADS;
    if(t > 512) t = 512; // matches alpha_scan_kernel __launch_bounds__
    if(t < 32) t = 32;
    // The deficit reduction is warp-based, so any multiple of 32 works now
    // (the old 512-slot reduction required a power of two).
    return t & ~31;
}

static int grid_blocks_for(uint64_t slice, int threads, int bpsm) {
    const uint64_t need = (slice + (uint64_t)threads - 1) / (uint64_t)threads;
    const uint64_t sat = (uint64_t)g_ctx.sm_count * (uint64_t)bpsm;
    uint64_t blocks64 = need < sat ? need : sat;
    if(blocks64 < 1) blocks64 = 1;
    if(blocks64 > 2147483647ULL) blocks64 = 2147483647ULL;
    return (int)blocks64;
}

} // namespace

std::array<uint8_t, 32> alphanumeric_cpu_hash92(
    uint32_t block_number,
    const std::array<uint8_t, 32>& previous_hash,
    uint64_t timestamp,
    uint64_t nonce,
    uint64_t difficulty,
    const std::array<uint8_t, 32>& merkle_root) {
    return alpha_ref::hash92_fields(block_number, previous_hash, timestamp,
                                    nonce, difficulty, merkle_root);
}

bool alphanumeric_cuda_list_devices() {
    int n = 0;
    if(hipGetDeviceCount(&n) != hipSuccess) return false;
    std::cout << "HIP AMD devices: " << n << "\n";
    for(int i = 0; i < n; ++i) {
        hipDeviceProp_t p{};
        hipGetDeviceProperties(&p, i);
        std::cout << "  [" << i << "] " << p.name << " gfx target " << p.gcnArchName << "\n";
    }
    return n > 0;
}

std::string alphanumeric_cuda_device_name(int device) {
    hipDeviceProp_t p{};
    if(hipGetDeviceProperties(&p, device) != hipSuccess) return "GPU";
    return std::string(p.name);
}

bool alphanumeric_cuda_reset(int device) {
    destroy_ctx_nothrow();

    // Clear the device runtime after a failed launch. hipDeviceReset also
    // releases any poisoned context state that a simple free/realloc would keep.
    hipSetDevice(device);
    hipError_t reset_err = hipDeviceReset();
    if(reset_err != hipSuccess) {
        std::cerr << "HIP reset error " << hipGetErrorString(reset_err) << "\n";
        return false;
    }
    return ensure_ready(device);
}

void alphanumeric_cuda_request_cancel() {
    std::lock_guard<std::mutex> lk(g_cancel_mu);
    const uint32_t gen = g_job_gen.fetch_add(1u, std::memory_order_acq_rel) + 1u;
    if(!g_ctx.ready) return;  // ensure_ready seeds the device copy from g_job_gen
    if(hipSetDevice(g_ctx.device) != hipSuccess) return;
    *g_ctx.h_gen_src = gen;
    // Best effort by design: if this write is late or lost, the running batch
    // simply finishes and the runner's job_seq staleness check still applies.
    hipMemcpyAsync(g_ctx.d_live_gen, g_ctx.h_gen_src, sizeof(uint32_t),
                    hipMemcpyHostToDevice, g_ctx.cancel_stream);
}

bool alphanumeric_cuda_hash92(
    int device,
    uint32_t block_number,
    const std::array<uint8_t, 32>& previous_hash,
    uint64_t timestamp,
    uint64_t nonce,
    uint64_t difficulty,
    const std::array<uint8_t, 32>& merkle_root,
    unsigned char out32[32]) {
    if(!ensure_ready(device)) return false;
    std::array<uint8_t, 32> dummy_target{};
    const AlphaJobPre jw = build_job_pre(block_number, previous_hash, timestamp,
                                         difficulty, merkle_root, dummy_target);
    alpha_hash_one_kernel<<<1, 1, 0, g_ctx.stream>>>(jw, nonce, g_ctx.d_hash);
    ALPHA_HIP_CK(hipGetLastError());
    ALPHA_HIP_CK(hipMemcpyAsync(out32, g_ctx.d_hash, 32, hipMemcpyDeviceToHost, g_ctx.stream));
    ALPHA_HIP_CK(hipStreamSynchronize(g_ctx.stream));
    return true;
}

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
    int threads_arg,
    int blocks_per_sm_arg) {
    result = AlphanumericCudaResult{};
    if(nonce_count == 0) return true;
    if(!ensure_ready(device)) return false;

    const AlphaJobPre jw = build_job_pre(block_number, previous_hash, timestamp,
                                         difficulty, merkle_root, target_be);
    const uint32_t job_gen = g_job_gen.load(std::memory_order_acquire);

    const int threads = clamp_threads(threads_arg);
    int bpsm = blocks_per_sm_arg > 0 ? blocks_per_sm_arg : CAPMINER_HIP_BLOCKS_PER_CU;
    if(bpsm < 1) bpsm = 1;
    if(bpsm > 256) bpsm = 256;

    // STAGE 4: async clear of {found, deficit} only; nonce/hash are gated by
    // found and never read stale. Everything below runs on the miner stream
    // with exactly one synchronize on the common path.
    ALPHA_HIP_CK(hipMemsetAsync(g_ctx.d_result, 0, RESULT_CLEAR_BYTES, g_ctx.stream));

    // STAGE 3: slice so each launch fits 32-bit index math. Normal batches
    // (<= 2^28 from the runner, <= 2^31 in general) are exactly one slice and
    // pay no extra synchronization.
    uint64_t remaining = nonce_count;
    uint64_t s = start_nonce;
    uint64_t enqueued = 0;
    bool found_early = false;
    while(remaining != 0 && !found_early) {
        const uint64_t slice = remaining < SLICE_MAX ? remaining : SLICE_MAX;
        const int blocks = grid_blocks_for(slice, threads, bpsm);
        alpha_scan_kernel<<<blocks, threads, 0, g_ctx.stream>>>(
            jw, s, (uint32_t)slice, job_gen, g_ctx.d_result, g_ctx.d_live_gen);
        ALPHA_HIP_CK(hipGetLastError());
        enqueued += slice;
        s += slice;             // u64 wrap is intentional and safe
        remaining -= slice;
        if(remaining != 0) {
            // Multi-slice request (> 2^31 nonces): peek between slices so a
            // found share stops us from queueing hundreds of ms of dead work.
            ALPHA_HIP_CK(hipMemcpyAsync(g_ctx.h_result, g_ctx.d_result,
                                          sizeof(DeviceScanResult),
                                          hipMemcpyDeviceToHost, g_ctx.stream));
            ALPHA_HIP_CK(hipStreamSynchronize(g_ctx.stream));
            if(g_ctx.h_result->found) found_early = true;
        }
    }

    ALPHA_HIP_CK(hipMemcpyAsync(g_ctx.h_result, g_ctx.d_result,
                                  sizeof(DeviceScanResult),
                                  hipMemcpyDeviceToHost, g_ctx.stream));
    ALPHA_HIP_CK(hipStreamSynchronize(g_ctx.stream));

    const DeviceScanResult& hr = *g_ctx.h_result;
    if(hr.found) {
        result.found = true;
        result.nonce = hr.nonce;
        std::memcpy(result.hash, hr.hash, 32);
    }
    // Honest completed-work count: what was enqueued minus what early-exiting
    // threads left undone. Equals nonce_count on every full scan.
    const uint64_t deficit = hr.deficit;
    result.hashes_scanned = enqueued >= deficit ? enqueued - deficit : 0;
    stats.hashes += result.hashes_scanned;
    return true;
}

AlphanumericBenchResult alphanumeric_cuda_benchmark_batch(
    int device, int threads_arg, int blocks_per_sm_arg,
    double seconds, uint64_t batch_nonces) {
    AlphanumericBenchResult br{};
    if(seconds <= 0.0) seconds = 3.0;
    if(batch_nonces == 0) batch_nonces = 1ull << 24;
    br.batch = batch_nonces;
    br.seconds_requested = seconds;
    if(!ensure_ready(device)) return br;

    std::array<uint8_t, 32> prev{};
    std::array<uint8_t, 32> merkle{};
    std::array<uint8_t, 32> target{}; // all zero: only an all-zero hash matches, full scan
    for(int i = 0; i < 32; ++i) {
        prev[i] = (uint8_t)(0x33 + i);
        merkle[i] = (uint8_t)(0x77 + i);
    }

    MinerStats stats;
    AlphanumericCudaResult r{};
    uint64_t start = 0;

    // Warmup.
    if(!alphanumeric_cuda_scan92(device, 1, prev, 1800000000ULL, 464ULL, merkle, target,
                                 start, batch_nonces, r, stats, threads_arg, blocks_per_sm_arg)) {
        return br;
    }
    start += batch_nonces;

    using clk = std::chrono::steady_clock;
    br.accounting_ok = true;
    const auto t0 = clk::now();
    double elapsed = 0.0;
    while(elapsed < seconds) {
        if(!alphanumeric_cuda_scan92(device, 1, prev, 1800000000ULL, 464ULL, merkle, target,
                                     start, batch_nonces, r, stats, threads_arg, blocks_per_sm_arg)) {
            return br;
        }
        // On-device accounting check for free: with an impossible target every
        // launch must report exactly the requested batch as completed.
        if(r.hashes_scanned != batch_nonces) br.accounting_ok = false;
        br.total_hashes += r.hashes_scanned;
        br.launches += 1;
        start += batch_nonces;
        elapsed = std::chrono::duration<double>(clk::now() - t0).count();
    }
    br.seconds = elapsed;
    if(elapsed > 0.0 && br.launches > 0) {
        br.hashes_per_sec = (double)br.total_hashes / elapsed;
        br.avg_batch_ms = 1000.0 * elapsed / (double)br.launches;
        br.launches_per_sec = (double)br.launches / elapsed;
        br.ok = true;
    }
    return br;
}

double alphanumeric_cuda_benchmark(int device, int threads_arg, int blocks_per_sm_arg, double seconds) {
    const AlphanumericBenchResult br =
        alphanumeric_cuda_benchmark_batch(device, threads_arg, blocks_per_sm_arg,
                                          seconds, 1ull << 24);
    return br.ok ? br.hashes_per_sec : 0.0;
}

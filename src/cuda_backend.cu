#include "cuda_backend.hpp"
#include <cuda_runtime.h>
#include <iostream>
#include <cstring>
#include <chrono>
#include <cstdio>
#include "cuda_whirlpool_device.cuh"   // plain_T0..T7, plain_RC, whirlpool_compress_cuda, cap_whirlpool80_cuda, enc64le_cuda, BCUDA

// ===========================================================================
// Tunables. Override at configure time with e.g.
//   -DCAPMINER_CUDA_THREADS=128  /  -DCAPMINER_CUDA_BLOCKS_PER_SM=16
// Then read the ptxas -v output (registers/spills) to pick the best combo.
// ===========================================================================
#ifndef CAPMINER_CUDA_THREADS
#define CAPMINER_CUDA_THREADS 256          // try 128 / 256 / 512
#endif
#ifndef CAPMINER_CUDA_BLOCKS_PER_SM
#define CAPMINER_CUDA_BLOCKS_PER_SM 24     // resident-block target for the grid-stride loop
#endif
// Occupancy lever: tells the compiler to fit at least this many blocks per SM,
// which caps registers/thread accordingly (65536 / (THREADS * MIN_BLOCKS)).
// The hot kernel measured 110 regs/thread -> only 2 blocks/SM (33% occupancy);
// 3 -> ~85 regs (50%), 4 -> 64 regs (67%). Higher may spill to local memory --
// check the startup kernel report and benchmark to pick the best. Pure codegen
// hint: the computed hash is identical, so accepted shares are unaffected.
#ifndef CAPMINER_CUDA_MIN_BLOCKS
#define CAPMINER_CUDA_MIN_BLOCKS 3
#endif

#define CUDA_CK(call) do { \
    cudaError_t _e = (call); \
    if(_e != cudaSuccess) { \
        std::cerr << "CUDA error " << cudaGetErrorString(_e) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
    } \
} while(0)

// ---------------------------------------------------------------------------
// Per-job constants (uploaded once per job by cuda_setup_job).
//   c_midstate    : Whirlpool state after compressing header bytes [0..63].
//   c_block1_w0   : little-endian u64 of header bytes [64..71].
//   c_block1_nbits: little-endian u32 of header bytes [72..75] (nbits).
//   c_target_w[4] : share target as four little-endian u64 words.
// All are read uniformly by every thread -> optimal constant-memory broadcast.
// ---------------------------------------------------------------------------
__constant__ unsigned long long c_midstate[8];
__constant__ unsigned long long c_block1_w0;
__constant__ unsigned int       c_block1_nbits;
__constant__ unsigned long long c_target_w[4];
// Precomputed Whirlpool round keys K_1..K_10 (the key schedule), computed once
// per job from the midstate. They are nonce-independent, so the per-nonce
// kernel no longer recomputes the key schedule -- it just uses these. Uniform
// per-thread reads => constant-memory broadcast. Layout: c_rk[r*8 + j] = K_{r+1}[j].
__constant__ unsigned long long c_rk[80];

// ---------------------------------------------------------------------------
// Whirlpool round over SHARED tables. Same arithmetic as whirlpool_compress_cuda
// in the .cuh (verified bit-identical), but reads the 8 T-tables from shared
// memory named `s_T`. Shared memory tolerates the per-thread divergent indices
// that __constant__ memory would serialize, which is the main speedup here.
// Each kernel using these macros must declare:  __shared__ unsigned long long s_T[8][256];
// ---------------------------------------------------------------------------
#define WELT_S(in, a,b,c,d,e,f,g,h) \
    ( s_T[0][BCUDA(in##a,0)] ^ s_T[1][BCUDA(in##b,1)] ^ \
      s_T[2][BCUDA(in##c,2)] ^ s_T[3][BCUDA(in##d,3)] ^ \
      s_T[4][BCUDA(in##e,4)] ^ s_T[5][BCUDA(in##f,5)] ^ \
      s_T[6][BCUDA(in##g,6)] ^ s_T[7][BCUDA(in##h,7)] )

#define WROUND_S(in,out,c0,c1,c2,c3,c4,c5,c6,c7) do { \
    out##0 = WELT_S(in,0,7,6,5,4,3,2,1) ^ (c0); \
    out##1 = WELT_S(in,1,0,7,6,5,4,3,2) ^ (c1); \
    out##2 = WELT_S(in,2,1,0,7,6,5,4,3) ^ (c2); \
    out##3 = WELT_S(in,3,2,1,0,7,6,5,4) ^ (c3); \
    out##4 = WELT_S(in,4,3,2,1,0,7,6,5) ^ (c4); \
    out##5 = WELT_S(in,5,4,3,2,1,0,7,6) ^ (c5); \
    out##6 = WELT_S(in,6,5,4,3,2,1,0,7) ^ (c6); \
    out##7 = WELT_S(in,7,6,5,4,3,2,1,0) ^ (c7); \
} while(0)

#define WMOV_S(d,s) do { \
    d##0=s##0; d##1=s##1; d##2=s##2; d##3=s##3; \
    d##4=s##4; d##5=s##5; d##6=s##6; d##7=s##7; \
} while(0)

// Cooperatively copy the 8 constant T-tables (16 KiB) into shared memory.
__device__ __forceinline__ void load_tables_shared(unsigned long long s_T[8][256]) {
    for(int i = threadIdx.x; i < 256; i += blockDim.x) {
        s_T[0][i] = plain_T0[i];
        s_T[1][i] = plain_T1[i];
        s_T[2][i] = plain_T2[i];
        s_T[3][i] = plain_T3[i];
        s_T[4][i] = plain_T4[i];
        s_T[5][i] = plain_T5[i];
        s_T[6][i] = plain_T6[i];
        s_T[7][i] = plain_T7[i];
    }
    __syncthreads();
}

// Build the second Whirlpool block words for `nonce`, run the single
// compression from the job midstate, and fold 512->256 into four LE words.
// (`s_T` must be in scope.) This is the exact per-nonce work, shared by the
// miner and the single-nonce validator so they can never diverge.
#define WHIRLPOOL_BLOCK1_FOLD(s_T, nonce, ow0, ow1, ow2, ow3) do {              \
    const unsigned long long _m0=c_midstate[0], _m1=c_midstate[1],             \
                             _m2=c_midstate[2], _m3=c_midstate[3],             \
                             _m4=c_midstate[4], _m5=c_midstate[5],             \
                             _m6=c_midstate[6], _m7=c_midstate[7];             \
    /* block1 words: [w0]=hdr[64..71] [w1]=nbits|nonce<<32 [w2]=0x80 [w7]=len */ \
    unsigned long long n0=c_block1_w0;                                         \
    unsigned long long n1=(unsigned long long)c_block1_nbits |                 \
                          ((unsigned long long)(nonce) << 32);                 \
    unsigned long long n2=0x80ULL, n3=0ULL, n4=0ULL, n5=0ULL, n6=0ULL;         \
    unsigned long long n7=0x8002000000000000ULL;                              \
    const unsigned long long sn0=n0, sn1=n1, sn2=n2, sn7=n7; /* sn3..6 == 0 */ \
    /* AddRoundKey K_0 (= midstate) */                                         \
    n0^=_m0; n1^=_m1; n2^=_m2; n3^=_m3; n4^=_m4; n5^=_m5; n6^=_m6; n7^=_m7;     \
    unsigned long long x0,x1,x2,x3,x4,x5,x6,x7;                                \
    /* Message-only rounds using the precomputed round keys K_1..K_10. The     \
       key schedule is NOT recomputed per nonce -- that is the optimization. */ \
    _Pragma("unroll")                                                          \
    for(int _r=0; _r<10; ++_r) {                                               \
        const unsigned long long *rk = &c_rk[_r*8];                            \
        WROUND_S(n,x, rk[0],rk[1],rk[2],rk[3],rk[4],rk[5],rk[6],rk[7]);        \
        WMOV_S(n,x);                                                           \
    }                                                                          \
    /* Miyaguchi-Preneel: state ^= block ^ message */                         \
    unsigned long long f0=_m0^n0^sn0, f1=_m1^n1^sn1, f2=_m2^n2^sn2,            \
                       f3=_m3^n3;       /* sn3==0 */                           \
    unsigned long long f4=_m4^n4, f5=_m5^n5, f6=_m6^n6, f7=_m7^n7^sn7;         \
    /* fold 64-byte digest halves: ow_j = f_j ^ f_{j+4}, output LE */          \
    (ow0)=f0^f4; (ow1)=f1^f5; (ow2)=f2^f6; (ow3)=f3^f7;                        \
} while(0)

// ---------------------------------------------------------------------------
// Mining kernel: grid-stride over [start_nonce, start_nonce+count).
// Compares hash <= target as a 256-bit little-endian integer.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(CAPMINER_CUDA_THREADS, CAPMINER_CUDA_MIN_BLOCKS)
whirlpool_mine_ms(uint32_t start_nonce, uint32_t count,
                  unsigned int* __restrict__ found,
                  uint32_t* __restrict__ found_nonce,
                  unsigned char* __restrict__ found_hash) {
    __shared__ unsigned long long s_T[8][256];
    load_tables_shared(s_T);

    const unsigned long long t0=c_target_w[0], t1=c_target_w[1],
                             t2=c_target_w[2], t3=c_target_w[3];

    const uint32_t stride = gridDim.x * blockDim.x;
    unsigned int poll = 0;
    for(uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += stride) {
        // Poll the global "found" flag only every 512 nonces. An uncached
        // volatile global read on every nonce is a major memory stall on the
        // hot path. Correctness is unchanged: the winner is still chosen by
        // atomicCAS below; other threads simply notice a win up to 512 nonces
        // later, which is negligible wasted work.
        if(((poll++ & 511u) == 0u) && ((const volatile unsigned int*)found)[0]) break;
        const uint32_t nonce = start_nonce + i;

        unsigned long long ow0, ow1, ow2, ow3;
        WHIRLPOOL_BLOCK1_FOLD(s_T, nonce, ow0, ow1, ow2, ow3);

        // hash <= target, most-significant word first
        bool le;
        if(ow3 != t3)      le = ow3 < t3;
        else if(ow2 != t2) le = ow2 < t2;
        else if(ow1 != t1) le = ow1 < t1;
        else               le = ow0 <= t0;

        if(le) {
            if(atomicCAS(found, 0u, 1u) == 0u) {
                *found_nonce = nonce;
                enc64le_cuda(found_hash + 0,  ow0);
                enc64le_cuda(found_hash + 8,  ow1);
                enc64le_cuda(found_hash + 16, ow2);
                enc64le_cuda(found_hash + 24, ow3);
            }
            break;
        }
    }
}

// Single-nonce probe: identical per-nonce math, writes the folded hash
// unconditionally. Used only for CPU<->GPU validation.
__global__ void whirlpool_probe_kernel(uint32_t nonce, unsigned char* __restrict__ out32) {
    __shared__ unsigned long long s_T[8][256];
    load_tables_shared(s_T);
    if(threadIdx.x == 0) {
        unsigned long long ow0, ow1, ow2, ow3;
        WHIRLPOOL_BLOCK1_FOLD(s_T, nonce, ow0, ow1, ow2, ow3);
        enc64le_cuda(out32 + 0,  ow0);
        enc64le_cuda(out32 + 8,  ow1);
        enc64le_cuda(out32 + 16, ow2);
        enc64le_cuda(out32 + 24, ow3);
    }
}

// Precompute the Whirlpool key schedule (round keys K_1..K_10) from the job
// midstate, once per job. Uses the SAME reference round function as the midstate
// kernel, so the keys are bit-identical to what the old per-nonce inline
// schedule produced. out_rk layout: [r*8 + j] = K_{r+1}[j].
__global__ void compute_round_keys_kernel(const unsigned long long* __restrict__ mid,
                                          unsigned long long* __restrict__ out_rk) {
    whirlpool_round_keys_cuda(mid, out_rk);
}

// Compute the block0 midstate once per job, reusing the reference compression
// so it is bit-identical to the first block of the full 80-byte hash.
__global__ void compute_midstate_kernel(const unsigned char* __restrict__ header80,
                                        unsigned long long* __restrict__ out_mid) {
    unsigned long long state[8] = {0,0,0,0,0,0,0,0};
    unsigned char block0[64];
    #pragma unroll
    for(int i=0;i<64;i++) block0[i] = header80[i];
    whirlpool_compress_cuda(block0, state);
    #pragma unroll
    for(int i=0;i<8;i++) out_mid[i] = state[i];
}

// ===========================================================================
// Host side
// ===========================================================================
namespace {

struct DevCtx {
    int            device      = -1;
    bool           ready       = false;   // buffers allocated
    bool           have_job    = false;   // constants uploaded
    int            sm_count    = 0;
    unsigned int*  d_found     = nullptr; // [1]
    uint32_t*      d_nonce     = nullptr; // [1]
    unsigned char* d_hash      = nullptr; // [32]
    unsigned char* d_hdr       = nullptr; // [80] (midstate input)
    unsigned long long* d_mid  = nullptr; // [8]
    unsigned long long* d_rk   = nullptr; // [80] precomputed round keys
};

DevCtx g_ctx;   // single mining device (scaffold uses --devices 0)

inline unsigned long long dec64le_host(const uint8_t* p) {
    unsigned long long v = 0;
    for(int i=0;i<8;i++) v |= (unsigned long long)p[i] << (8*i);
    return v;
}

bool ensure_ready(int device) {
    if(g_ctx.ready && g_ctx.device == device) {
        CUDA_CK(cudaSetDevice(device));
        return true;
    }
    // (Re)initialize for this device.
    if(g_ctx.ready) {
        cudaFree(g_ctx.d_found); cudaFree(g_ctx.d_nonce); cudaFree(g_ctx.d_hash);
        cudaFree(g_ctx.d_hdr);   cudaFree(g_ctx.d_mid);   cudaFree(g_ctx.d_rk);
        g_ctx = DevCtx{};
    }
    CUDA_CK(cudaSetDevice(device));
    g_ctx.device = device;
    g_ctx.sm_count = 1;
    cudaDeviceGetAttribute(&g_ctx.sm_count, cudaDevAttrMultiProcessorCount, device);
    if(g_ctx.sm_count < 1) g_ctx.sm_count = 1;

    CUDA_CK(cudaMalloc(&g_ctx.d_found, sizeof(unsigned int)));
    CUDA_CK(cudaMalloc(&g_ctx.d_nonce, sizeof(uint32_t)));
    CUDA_CK(cudaMalloc(&g_ctx.d_hash,  32));
    CUDA_CK(cudaMalloc(&g_ctx.d_hdr,   80));
    CUDA_CK(cudaMalloc(&g_ctx.d_mid,   8 * sizeof(unsigned long long)));
    CUDA_CK(cudaMalloc(&g_ctx.d_rk,   80 * sizeof(unsigned long long)));
    g_ctx.ready = true;
    g_ctx.have_job = false;

    // The hot kernel uses 16 KiB of static shared memory for the T-tables, which
    // caps resident blocks per SM. Prefer the largest shared-memory carveout so
    // more blocks (more warps) stay resident to hide the table-lookup latency.
    cudaFuncSetAttribute(whirlpool_mine_ms,
                         cudaFuncAttributePreferredSharedMemoryCarveout,
                         cudaSharedmemCarveoutMaxShared);
    return true;
}

} // namespace

bool cuda_list_devices() {
    int n=0;
    if(cudaGetDeviceCount(&n)!=cudaSuccess) return false;
    std::cout << "CUDA devices: " << n << "\n";
    for(int i=0;i<n;i++) {
        cudaDeviceProp p{};
        cudaGetDeviceProperties(&p,i);
        std::cout << "  [" << i << "] " << p.name << " SM " << p.major << "." << p.minor << "\n";
    }
    return n>0;
}

std::string cuda_device_name(int device) {
    cudaDeviceProp p{};
    if(cudaGetDeviceProperties(&p, device) != cudaSuccess) return "GPU";
    return std::string(p.name);
}

bool cuda_setup_job(int device,
                    const std::array<uint8_t, 80>& header80,
                    const std::array<uint8_t, 32>& target) {
    if(!ensure_ready(device)) return false;

    // Midstate from block0 (header[0..63]) on the GPU.
    CUDA_CK(cudaMemcpy(g_ctx.d_hdr, header80.data(), 80, cudaMemcpyHostToDevice));
    compute_midstate_kernel<<<1,1>>>(g_ctx.d_hdr, g_ctx.d_mid);
    CUDA_CK(cudaGetLastError());

    unsigned long long mid[8];
    CUDA_CK(cudaMemcpy(mid, g_ctx.d_mid, sizeof(mid), cudaMemcpyDeviceToHost));
    CUDA_CK(cudaMemcpyToSymbol(c_midstate, mid, sizeof(mid)));

    // Precompute the key schedule (round keys K_1..K_10) once per job and upload
    // to constant memory; the per-nonce kernel then skips the schedule entirely.
    compute_round_keys_kernel<<<1,1>>>(g_ctx.d_mid, g_ctx.d_rk);
    CUDA_CK(cudaGetLastError());
    unsigned long long rk[80];
    CUDA_CK(cudaMemcpy(rk, g_ctx.d_rk, sizeof(rk), cudaMemcpyDeviceToHost));
    CUDA_CK(cudaMemcpyToSymbol(c_rk, rk, sizeof(rk)));

    // header tail: w0 = bytes[64..71], nbits = bytes[72..75]
    unsigned long long w0 = dec64le_host(header80.data() + 64);
    unsigned int nbits = (unsigned int)(dec64le_host(header80.data() + 72) & 0xffffffffULL);
    CUDA_CK(cudaMemcpyToSymbol(c_block1_w0,    &w0,    sizeof(w0)));
    CUDA_CK(cudaMemcpyToSymbol(c_block1_nbits, &nbits, sizeof(nbits)));

    // target as four little-endian u64 words
    unsigned long long tw[4];
    for(int j=0;j<4;j++) tw[j] = dec64le_host(target.data() + 8*j);
    CUDA_CK(cudaMemcpyToSymbol(c_target_w, tw, sizeof(tw)));

    CUDA_CK(cudaDeviceSynchronize());
    g_ctx.have_job = true;
    return true;
}

bool cuda_hash_one_nonce(int device, uint32_t nonce, unsigned char out32[32]) {
    if(!g_ctx.ready || g_ctx.device != device || !g_ctx.have_job) return false;
    CUDA_CK(cudaSetDevice(device));
    CUDA_CK(cudaMemset(g_ctx.d_hash, 0, 32));
    whirlpool_probe_kernel<<<1, CAPMINER_CUDA_THREADS>>>(nonce, g_ctx.d_hash);
    CUDA_CK(cudaGetLastError());
    CUDA_CK(cudaDeviceSynchronize());
    CUDA_CK(cudaMemcpy(out32, g_ctx.d_hash, 32, cudaMemcpyDeviceToHost));
    return true;
}

bool cuda_mine_batch(int device, uint32_t start_nonce, uint32_t count,
                     CudaResult& result, MinerStats& stats,
                     int threads_arg, int blocks_per_sm_arg) {
    if(count == 0) return true;
    if(!g_ctx.ready || g_ctx.device != device || !g_ctx.have_job) {
        std::cerr << "cuda_mine_batch called before cuda_setup_job\n";
        return false;
    }
    CUDA_CK(cudaSetDevice(device));
    CUDA_CK(cudaMemset(g_ctx.d_found, 0, sizeof(unsigned int)));

    // Threads per block: runtime override, else compiled default. The kernel is
    // compiled with __launch_bounds__(CAPMINER_CUDA_THREADS), so launching with
    // more than that fails -- clamp to it. Keep a warp multiple.
    int threads = threads_arg > 0 ? threads_arg : CAPMINER_CUDA_THREADS;
    if(threads > CAPMINER_CUDA_THREADS) threads = CAPMINER_CUDA_THREADS;
    if(threads < 32) threads = 32;
    threads = (threads / 32) * 32;
    if(threads < 32) threads = 32;

    int bpsm = blocks_per_sm_arg > 0 ? blocks_per_sm_arg : CAPMINER_CUDA_BLOCKS_PER_SM;
    if(bpsm < 1) bpsm = 1;

    // Saturate the GPU; the grid-stride loop covers all `count` nonces while
    // the 16 KiB shared-table load is amortized over many hashes per block.
    long long sat   = (long long)g_ctx.sm_count * bpsm;
    long long need  = ((long long)count + threads - 1) / threads;
    long long blksL = need < sat ? need : sat;
    if(blksL < 1) blksL = 1;
    int blocks = (int)blksL;

    whirlpool_mine_ms<<<blocks, threads>>>(start_nonce, count,
                                           g_ctx.d_found, g_ctx.d_nonce, g_ctx.d_hash);
    CUDA_CK(cudaGetLastError());
    CUDA_CK(cudaDeviceSynchronize());

    unsigned int found = 0;
    CUDA_CK(cudaMemcpy(&found, g_ctx.d_found, sizeof(found), cudaMemcpyDeviceToHost));
    if(found) {
        result.found = true;
        CUDA_CK(cudaMemcpy(&result.nonce, g_ctx.d_nonce, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CK(cudaMemcpy(result.hash,   g_ctx.d_hash,  32,               cudaMemcpyDeviceToHost));
    }
    stats.hashes += count;
    return true;
}

// --- Reference full 80-byte path (kept for cross-checking) -----------------
__global__ void whirlpool_hash_once_kernel(const unsigned char* __restrict__ header80,
                                           unsigned char* __restrict__ out32) {
    unsigned char h[80];
    #pragma unroll
    for(int i=0;i<80;i++) h[i]=header80[i];
    cap_whirlpool80_cuda(h, out32);
}

bool cuda_hash_once(int device, const std::array<uint8_t, 80>& header80, unsigned char out32[32]) {
    if(!ensure_ready(device)) return false;
    CUDA_CK(cudaMemcpy(g_ctx.d_hdr, header80.data(), 80, cudaMemcpyHostToDevice));
    CUDA_CK(cudaMemset(g_ctx.d_hash, 0, 32));
    whirlpool_hash_once_kernel<<<1,1>>>(g_ctx.d_hdr, g_ctx.d_hash);
    CUDA_CK(cudaGetLastError());
    CUDA_CK(cudaDeviceSynchronize());
    CUDA_CK(cudaMemcpy(out32, g_ctx.d_hash, 32, cudaMemcpyDeviceToHost));
    return true;
}

// ===========================================================================
// Diagnostics & benchmark (no effect on the hash; throughput tooling only)
// ===========================================================================
static int clamp_threads(int t) {
    if(t <= 0) t = CAPMINER_CUDA_THREADS;
    if(t > CAPMINER_CUDA_THREADS) t = CAPMINER_CUDA_THREADS;  // launch-bound cap
    if(t < 32) t = 32;
    t = (t / 32) * 32;
    return t < 32 ? 32 : t;
}

void cuda_print_kernel_info(int device, int threads) {
    if(!ensure_ready(device)) return;
    threads = clamp_threads(threads);

    cudaFuncAttributes a{};
    if(cudaFuncGetAttributes(&a, whirlpool_mine_ms) != cudaSuccess) return;

    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, whirlpool_mine_ms, threads, 0);

    cudaDeviceProp p{};
    cudaGetDeviceProperties(&p, device);
    int warps_per_sm   = blocks_per_sm * (threads / 32);
    int max_warps_per_sm = p.maxThreadsPerMultiProcessor / 32;
    double occ = max_warps_per_sm > 0 ? (100.0 * warps_per_sm / max_warps_per_sm) : 0.0;

    std::printf("kernel whirlpool_mine_ms @ %d threads/block:\n", threads);
    std::printf("  registers/thread : %d\n", a.numRegs);
    std::printf("  shared/block     : %zu bytes\n", (size_t)a.sharedSizeBytes);
    std::printf("  local/spill      : %zu bytes/thread\n", (size_t)a.localSizeBytes);
    std::printf("  max threads/block: %d\n", a.maxThreadsPerBlock);
    std::printf("  SM count         : %d\n", p.multiProcessorCount);
    std::printf("  resident blocks/SM: %d  -> warps/SM %d of %d  (occupancy ~%.0f%%)\n",
                blocks_per_sm, warps_per_sm, max_warps_per_sm, occ);
    if(a.localSizeBytes > 0)
        std::printf("  NOTE: register spilling to local memory detected (%zu B/thread)\n",
                    (size_t)a.localSizeBytes);
    std::fflush(stdout);
}

double cuda_benchmark(int device, int threads_arg, int blocks_per_sm_arg, double seconds) {
    if(!ensure_ready(device)) return 0.0;
    CUDA_CK(cudaSetDevice(device));

    // Synthetic job: arbitrary midstate, target = 0 so nothing ever matches and
    // the kernel hashes the entire range each launch (pure throughput, no early
    // exit, no atomics). Identical per-nonce work to real mining.
    unsigned long long mid[8] = {
        0x0123456789abcdefULL, 0xfedcba9876543210ULL, 0x1111111111111111ULL,
        0x2222222222222222ULL, 0x3333333333333333ULL, 0x4444444444444444ULL,
        0x5555555555555555ULL, 0x6666666666666666ULL };
    unsigned long long w0 = 0x89abcdef01234567ULL;
    unsigned int nbits = 0x1d00ffffu;
    unsigned long long tw[4] = {0,0,0,0};
    CUDA_CK(cudaMemcpyToSymbol(c_midstate,    mid, sizeof(mid)));
    CUDA_CK(cudaMemcpyToSymbol(c_block1_w0,    &w0, sizeof(w0)));
    CUDA_CK(cudaMemcpyToSymbol(c_block1_nbits, &nbits, sizeof(nbits)));
    CUDA_CK(cudaMemcpyToSymbol(c_target_w,     tw, sizeof(tw)));
    // Round keys: arbitrary values (throughput is independent of their value).
    unsigned long long rk[80];
    for(int i=0;i<80;i++) rk[i] = 0x0123456789abcdefULL * (unsigned long long)(i+1);
    CUDA_CK(cudaMemcpyToSymbol(c_rk, rk, sizeof(rk)));
    g_ctx.have_job = true;

    int threads = clamp_threads(threads_arg);
    int bpsm = blocks_per_sm_arg > 0 ? blocks_per_sm_arg : CAPMINER_CUDA_BLOCKS_PER_SM;
    if(bpsm < 1) bpsm = 1;

    const uint32_t count = 1u << 27;                 // 134M nonces per launch
    long long sat   = (long long)g_ctx.sm_count * bpsm;
    long long need  = ((long long)count + threads - 1) / threads;
    int blocks = (int)(need < sat ? need : sat);
    if(blocks < 1) blocks = 1;

    // Warm up (also forces JIT / cache fill).
    CUDA_CK(cudaMemset(g_ctx.d_found, 0, sizeof(unsigned int)));
    whirlpool_mine_ms<<<blocks, threads>>>(0, count, g_ctx.d_found, g_ctx.d_nonce, g_ctx.d_hash);
    CUDA_CK(cudaDeviceSynchronize());

    using clk = std::chrono::steady_clock;
    auto t0 = clk::now();
    unsigned long long total = 0;
    uint32_t start = 0;
    double elapsed = 0.0;
    while(elapsed < seconds) {
        CUDA_CK(cudaMemset(g_ctx.d_found, 0, sizeof(unsigned int)));
        whirlpool_mine_ms<<<blocks, threads>>>(start, count, g_ctx.d_found, g_ctx.d_nonce, g_ctx.d_hash);
        CUDA_CK(cudaDeviceSynchronize());
        total += count;
        start += count;
        elapsed = std::chrono::duration<double>(clk::now() - t0).count();
    }
    double hps = elapsed > 0 ? (double)total / elapsed : 0.0;
    std::printf("  threads=%-4d blocks/SM=%-3d grid=%-6d -> %.3f GH/s (%.0f MH/s)\n",
                threads, bpsm, blocks, hps / 1e9, hps / 1e6);
    std::fflush(stdout);
    return hps;
}

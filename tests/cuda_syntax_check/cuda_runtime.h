// =============================================================================
// tests/cuda_syntax_check/cuda_runtime.h
//
// SYNTAX/TYPE-CHECK SHIM ONLY. Lets a plain host C++ compiler parse and
// type-check alphanumeric_cuda_backend.cu when nvcc is unavailable (the
// <<<...>>> launches are rewritten to ALPHA_SHIM_LAUNCH by check.sh first).
// Nothing here executes real GPU work; it exists purely so a typo in the .cu
// cannot survive to the real Windows/nvcc build.
// =============================================================================
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline
#define __restrict__
#define __launch_bounds__(...)
#define __shared__ static
template <class... Ts> inline int ALPHA_SHIM_CFG(Ts&&...) { return 0; }

using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
using cudaStream_t = void*;
constexpr unsigned cudaStreamNonBlocking = 1;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice };
enum { cudaDevAttrMultiProcessorCount = 16 };

struct cudaDeviceProp { char name[256]; int major; int minor; };
struct alpha_dim3 { unsigned x, y, z; };
inline alpha_dim3 threadIdx{0,0,0}, blockIdx{0,0,0}, blockDim{1,1,1}, gridDim{1,1,1};

inline const char* cudaGetErrorString(cudaError_t) { return "shim"; }
inline cudaError_t cudaSetDevice(int) { return cudaSuccess; }
inline cudaError_t cudaGetDeviceCount(int* n) { *n = 0; return cudaSuccess; }
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) { std::memset(p, 0, sizeof(*p)); return cudaSuccess; }
inline cudaError_t cudaDeviceGetAttribute(int* v, int, int) { *v = 1; return cudaSuccess; }
template <class T> cudaError_t cudaMalloc(T** p, size_t) { *p = nullptr; return cudaSuccess; }
inline cudaError_t cudaFree(void*) { return cudaSuccess; }
template <class T> cudaError_t cudaMallocHost(T** p, size_t) { *p = nullptr; return cudaSuccess; }
inline cudaError_t cudaFreeHost(void*) { return cudaSuccess; }
inline cudaError_t cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind) { return cudaSuccess; }
inline cudaError_t cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaMemsetAsync(void*, int, size_t, cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) { *s = nullptr; return cudaSuccess; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { return cudaSuccess; }
inline cudaError_t cudaDeviceReset() { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }

inline void __syncthreads() {}
inline unsigned atomicCAS(unsigned* p, unsigned cmp, unsigned val) {
    const unsigned old = *p;
    if(old == cmp) *p = val;
    return old;
}
inline unsigned long long atomicAdd(unsigned long long* p, unsigned long long v) {
    const unsigned long long old = *p;
    *p += v;
    return old;
}
inline uint32_t __shfl_down_sync(unsigned, uint32_t v, int) { return v; }
template <class T> T __ldcg(const T* p) { return *p; }
inline uint32_t __funnelshift_r(uint32_t lo, uint32_t hi, uint32_t s) {
    (void)hi;
    return (lo >> (s & 31)) | (lo << ((32 - (s & 31)) & 31));
}
inline uint32_t __byte_perm(uint32_t a, uint32_t, uint32_t) {
    return (a >> 24) | ((a >> 8) & 0x0000FF00u) | ((a << 8) & 0x00FF0000u) | (a << 24);
}

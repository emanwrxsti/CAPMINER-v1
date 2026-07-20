#include "cuda_backend.hpp"
#include <cstring>

bool cuda_list_devices() { return false; }
std::string cuda_device_name(int) { return "CUDA disabled"; }
bool cuda_setup_job(int, const std::array<uint8_t, 80>&,
                    const std::array<uint8_t, 32>&) { return false; }
bool cuda_hash_one_nonce(int, uint32_t, unsigned char out32[32]) {
    std::memset(out32, 0, 32);
    return false;
}
bool cuda_mine_batch(int, uint32_t, uint32_t, CudaResult& result,
                     MinerStats&, int, int) {
    result = CudaResult{};
    return false;
}
bool cuda_hash_once(int, const std::array<uint8_t, 80>&, unsigned char out32[32]) {
    std::memset(out32, 0, 32);
    return false;
}
void cuda_print_kernel_info(int, int) {}
double cuda_benchmark(int, int, int, double) { return 0.0; }

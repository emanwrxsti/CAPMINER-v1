#include "nvml_telemetry.hpp"

#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
  #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
#else
  #include <dlfcn.h>
#endif

#ifndef NVML_SUCCESS
#define NVML_SUCCESS 0
#endif
#define CAPMINER_NVML_TEMPERATURE_GPU 0

namespace {
#ifdef _WIN32
void* open_nvml() {
    HMODULE h = LoadLibraryA("nvml.dll");
    if(!h) h = LoadLibraryA("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
    return reinterpret_cast<void*>(h);
}
void* load_symbol(void* lib, const char* name) {
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
}
void close_library(void* lib) {
    if(lib) FreeLibrary(reinterpret_cast<HMODULE>(lib));
}
#else
void* open_nvml() {
    void* h = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
    if(!h) h = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_LOCAL);
    return h;
}
void* load_symbol(void* lib, const char* name) { return dlsym(lib, name); }
void close_library(void* lib) { if(lib) dlclose(lib); }
#endif
}

NvmlTelemetry::~NvmlTelemetry() {
    if(pShutdown_ && available_) pShutdown_();
    close_library(lib_);
}

bool NvmlTelemetry::init(unsigned int device_index) {
    if(available_) return true;
    void* h = open_nvml();
    if(!h) return false;
    lib_ = h;

    auto initFn = reinterpret_cast<nvmlInit_t>(load_symbol(h, "nvmlInit_v2"));
    if(!initFn) initFn = reinterpret_cast<nvmlInit_t>(load_symbol(h, "nvmlInit"));
    auto getHandle = reinterpret_cast<nvmlDeviceGetHandleByIndex_t>(
        load_symbol(h, "nvmlDeviceGetHandleByIndex_v2"));
    if(!getHandle) {
        getHandle = reinterpret_cast<nvmlDeviceGetHandleByIndex_t>(
            load_symbol(h, "nvmlDeviceGetHandleByIndex"));
    }

    pShutdown_ = reinterpret_cast<nvmlShutdown_t>(load_symbol(h, "nvmlShutdown"));
    pPower_ = reinterpret_cast<nvmlDeviceGetPowerUsage_t>(
        load_symbol(h, "nvmlDeviceGetPowerUsage"));
    pTemp_ = reinterpret_cast<nvmlDeviceGetTemperature_t>(
        load_symbol(h, "nvmlDeviceGetTemperature"));
    pFan_ = reinterpret_cast<nvmlDeviceGetFanSpeed_t>(
        load_symbol(h, "nvmlDeviceGetFanSpeed"));

    if(!initFn || !getHandle || initFn() != NVML_SUCCESS) {
        close_library(lib_);
        lib_ = nullptr;
        return false;
    }

    nvmlDevice_handle dev = nullptr;
    if(getHandle(device_index, &dev) != NVML_SUCCESS || !dev) {
        if(pShutdown_) pShutdown_();
        close_library(lib_);
        lib_ = nullptr;
        return false;
    }

    dev_ = dev;
    available_ = true;
    return true;
}

GpuTelemetry NvmlTelemetry::sample() {
    GpuTelemetry t;
    if(!available_) return t;
    unsigned int v = 0;
    if(pPower_ && pPower_(dev_, &v) == NVML_SUCCESS) {
        t.has_power = true;
        t.power_mw = v;
    }
    v = 0;
    if(pTemp_ && pTemp_(dev_, CAPMINER_NVML_TEMPERATURE_GPU, &v) == NVML_SUCCESS) {
        t.has_temp = true;
        t.temp_c = v;
    }
    v = 0;
    if(pFan_ && pFan_(dev_, &v) == NVML_SUCCESS) {
        t.has_fan = true;
        t.fan_pct = v;
    }
    return t;
}

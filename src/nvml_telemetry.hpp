#pragma once
#include <string>

// ---------------------------------------------------------------------------
// Optional GPU telemetry via NVIDIA NVML, loaded *dynamically* at runtime.
//
// nvml.dll ships with the NVIDIA driver. We never link against it and never
// require it: if it is missing, or any query fails, telemetry is simply
// reported as unavailable and the miner keeps running normally.
//
// This is read-only telemetry (power draw, temperature, fan speed). It does
// not change clocks, power limits, or anything else on the GPU.
// ---------------------------------------------------------------------------
struct GpuTelemetry {
    bool         has_power = false;  unsigned int power_mw = 0;   // milliwatts
    bool         has_temp  = false;  unsigned int temp_c   = 0;   // Celsius
    bool         has_fan   = false;  unsigned int fan_pct  = 0;   // percent
};

// NVML's device handle is an opaque pointer to an incomplete struct; void* matches its ABI.
typedef void* nvmlDevice_handle;

// Minimal NVML entry-point signatures, declared here so we don't need nvml.h.
typedef int (*nvmlInit_t)(void);
typedef int (*nvmlShutdown_t)(void);
typedef int (*nvmlDeviceGetHandleByIndex_t)(unsigned int, nvmlDevice_handle*);
typedef int (*nvmlDeviceGetPowerUsage_t)(nvmlDevice_handle, unsigned int*);
typedef int (*nvmlDeviceGetTemperature_t)(nvmlDevice_handle, int, unsigned int*);
typedef int (*nvmlDeviceGetFanSpeed_t)(nvmlDevice_handle, unsigned int*);

class NvmlTelemetry {
public:
    NvmlTelemetry() = default;
    ~NvmlTelemetry();

    NvmlTelemetry(const NvmlTelemetry&) = delete;
    NvmlTelemetry& operator=(const NvmlTelemetry&) = delete;

    // Try to load nvml.dll and obtain a handle for the given device index.
    // Returns true if NVML is usable. On failure, available() stays false and
    // the caller should keep going (telemetry just shows N/A).
    bool init(unsigned int device_index);
    bool available() const { return available_; }

    // Sample current values. Fields NVML can't provide keep has_*=false.
    GpuTelemetry sample();

private:
    bool                       available_ = false;
    void*                      lib_       = nullptr;   // HMODULE
    nvmlDevice_handle          dev_       = nullptr;
    nvmlShutdown_t             pShutdown_ = nullptr;
    nvmlDeviceGetPowerUsage_t  pPower_    = nullptr;
    nvmlDeviceGetTemperature_t pTemp_     = nullptr;
    nvmlDeviceGetFanSpeed_t    pFan_      = nullptr;
};

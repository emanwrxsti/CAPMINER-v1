cmake_minimum_required(VERSION 3.24)
project(capminer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

option(CAPMINER_ENABLE_CUDA "Build NVIDIA CUDA backend" ON)
option(CAPMINER_ENABLE_HIP "Build AMD HIP/ROCm Alphanumeric backend" OFF)
option(CAPMINER_ENABLE_OPENCL "Build legacy OpenCL platform detection" OFF)
set(CAPMINER_CUDA_ARCHITECTURES "75;86;89;120" CACHE STRING
    "CUDA SM architectures to compile, e.g. 75;86;89;120")
set(CAPMINER_HIP_ARCHITECTURES "gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201" CACHE STRING
    "AMD GPU targets for HIP, e.g. gfx1030;gfx1100;gfx1101;gfx1201")

if(CAPMINER_ENABLE_CUDA AND CAPMINER_ENABLE_HIP)
  message(FATAL_ERROR "Build CUDA and HIP as separate binaries; enable only one backend per build directory.")
endif()

find_package(Threads REQUIRED)

add_executable(capminer
  src/main.cpp
  src/config.cpp
  src/logger.cpp
  src/stratum_client.cpp
  src/stats.cpp
  src/share_validator.cpp
  src/nvml_telemetry.cpp

  src/whirlpool/whirlpool.cpp
  src/whirlpool/capstash_job.cpp
  src/whirlpool/sha256_simple.cpp
  src/whirlpool/capstash_pow.cpp
  src/alphanumeric/alphanumeric_runner.cpp

  src/crypto/whirlpool.cpp
  src/crypto/whirlpool/whirlpool.c
)

target_include_directories(capminer PRIVATE src)
target_link_libraries(capminer PRIVATE Threads::Threads)

if(WIN32)
  target_compile_definitions(capminer PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
  target_link_libraries(capminer PRIVATE ws2_32)
else()
  target_compile_definitions(capminer PRIVATE _FILE_OFFSET_BITS=64)
  target_link_libraries(capminer PRIVATE ${CMAKE_DL_LIBS})
endif()

if(MSVC)
  target_compile_options(capminer PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:/W4>)
else()
  target_compile_options(capminer PRIVATE $<$<COMPILE_LANGUAGE:C,CXX>:-Wall;-Wextra;-Wpedantic>)
endif()

if(CAPMINER_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_EXTENSIONS OFF)

  target_sources(capminer PRIVATE
    src/cuda_backend.cu
    src/alphanumeric/alphanumeric_cuda_backend.cu
  )
  target_compile_definitions(capminer PRIVATE CAPMINER_CUDA=1 CAPMINER_GPU_BACKEND_CUDA=1)
  set_target_properties(capminer PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    CUDA_ARCHITECTURES "${CAPMINER_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_RUNTIME_LIBRARY Static
  )
  target_compile_options(capminer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ptxas-options=-v>
  )
elseif(CAPMINER_ENABLE_HIP)
  if(WIN32)
    message(FATAL_ERROR "CMake HIP language is not supported by AMD HIP SDK on Windows. Run BUILD_WINDOWS_AMD.bat instead.")
  endif()
  enable_language(HIP)
  set(CMAKE_HIP_STANDARD 17)
  set(CMAKE_HIP_STANDARD_REQUIRED ON)
  set(CMAKE_HIP_EXTENSIONS OFF)

  # AMD support currently targets the Alphanumeric BLAKE3-92 engine. The
  # CapStash Whirlpool backend remains NVIDIA-only, so link its host stub.
  target_sources(capminer PRIVATE
    src/cuda_backend_stub.cpp
    src/alphanumeric/alphanumeric_hip_backend.hip
  )
  target_compile_definitions(capminer PRIVATE CAPMINER_HIP=1 CAPMINER_GPU_BACKEND_HIP=1)
  set_target_properties(capminer PROPERTIES
    HIP_STANDARD 17
    HIP_STANDARD_REQUIRED ON
    HIP_EXTENSIONS OFF
    HIP_ARCHITECTURES "${CAPMINER_HIP_ARCHITECTURES}"
  )
else()
  target_sources(capminer PRIVATE
    src/cuda_backend_stub.cpp
    src/alphanumeric/alphanumeric_cuda_stub.cpp
  )
endif()

if(CAPMINER_ENABLE_OPENCL)
  find_package(OpenCL QUIET)
  if(OpenCL_FOUND)
    target_sources(capminer PRIVATE src/opencl_backend.cpp)
    target_link_libraries(capminer PRIVATE OpenCL::OpenCL)
    target_compile_definitions(capminer PRIVATE CAPMINER_OPENCL=1)
  else()
    message(WARNING "OpenCL not found. Building without OpenCL.")
  endif()
endif()

include(GNUInstallDirs)
install(TARGETS capminer RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

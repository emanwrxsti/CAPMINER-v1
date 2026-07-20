#include "logger.hpp"
#ifdef CAPMINER_OPENCL
#include <CL/cl.h>
#endif

bool opencl_backend_available() {
#ifdef CAPMINER_OPENCL
    cl_uint platforms=0;
    clGetPlatformIDs(0,nullptr,&platforms);
    log_line("OpenCL platforms: " + std::to_string(platforms));
    return platforms > 0;
#else
    log_line("OpenCL disabled at build time");
    return false;
#endif
}

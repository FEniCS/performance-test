#pragma once

#include <cstdio>
#include <sstream>

// Some useful utilities for error checking and synchronisation
// for each hardware type

#ifdef USE_HIP
#include <hip/hip_runtime.h>
#define err_check(command)                                                                         \
  {                                                                                                \
    hipError_t status = command;                                                                   \
    if (status != hipSuccess)                                                                      \
    {                                                                                              \
      printf("(%s:%d) Error: Hip reports %s\n", __FILE__, __LINE__, hipGetErrorString(status));    \
      exit(1);                                                                                     \
    }                                                                                              \
  }
#elif USE_CUDA
#include <cuda_runtime.h>
#define err_check(command)                                                                         \
  {                                                                                                \
    cudaError_t status = command;                                                                  \
    if (status != cudaSuccess)                                                                     \
    {                                                                                              \
      printf("(%s:%d) Error: CUDA reports %s\n", __FILE__, __LINE__, cudaGetErrorString(status));  \
      exit(1);                                                                                     \
    }                                                                                              \
  }
#endif

#ifdef USE_HIP
#define non_temp_load(addr) __builtin_nontemporal_load(addr)
#define deviceMemcpyToSymbol(symbol, addr, count) hipMemcpyToSymbol(symbol, addr, count)
void check_device_last_error() { err_check(hipGetLastError()); }
void device_synchronize() { err_check(hipDeviceSynchronize()); }
#elif USE_CUDA
#define non_temp_load(addr) __ldg(addr)
#define deviceMemcpyToSymbol(symbol, addr, count) cudaMemcpyToSymbol(symbol, addr, count)
void check_device_last_error() { err_check(cudaGetLastError()); }
void device_synchronize() { err_check(cudaDeviceSynchronize()); }
#else
#error "Unsupported platform"
#endif

std::string device_information()
{
  std::stringstream s;
  const int kb = 1024;
  const int mb = kb * kb;
  int devCount;

#ifdef USE_HIP
  hipError_t status = hipGetDeviceCount(&devCount);
  s << "Num devices: " << devCount << std::endl;
  hipDeviceProp_t props;
  status = hipGetDeviceProperties(&props, 0);
  if (status != hipSuccess)
    throw std::runtime_error("Error getting device properties");
  s << "Device: " << props.name << "/" << props.gcnArchName << ": " << props.major << "."
    << props.minor << std::endl;
#elif USE_CUDA
  cudaError_t status = cudaGetDeviceCount(&devCount);
  s << "Num devices: " << devCount << std::endl;
  cudaDeviceProp props;
  status = cudaGetDeviceProperties(&props, 0);
  if (status != cudaSuccess)
    throw std::runtime_error("Error getting device properties");
  s << "Device: " << props.name << ": " << props.major << "." << props.minor << std::endl;
#endif

  s << "  Global memory:   " << props.totalGlobalMem / mb << " Mb" << std::endl;
  s << "  Shared memory:   " << props.sharedMemPerBlock / kb << " kb" << std::endl;
  s << "  Constant memory: " << props.totalConstMem / mb << " Mb" << std::endl;
  s << "  Block registers: " << props.regsPerBlock << std::endl << std::endl;
  s << "  Warp size:         " << props.warpSize << std::endl;
  s << "  Threads per block: " << props.maxThreadsPerBlock << std::endl;
  s << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]
    << ", " << props.maxThreadsDim[2] << " ]" << std::endl;
  s << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", "
    << props.maxGridSize[2] << " ]" << std::endl;

  return s.str();
}

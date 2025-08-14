// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#include "util.hpp"

#if defined(USE_CUDA) || defined(USE_HIP)

//----------------------------------------------------------------------------
std::string benchdolfinx::get_device_information()
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
  s << "Device: " << props.name << "/" << props.gcnArchName << ": "
    << props.major << "." << props.minor << std::endl;
#elif USE_CUDA
  cudaError_t status = cudaGetDeviceCount(&devCount);
  s << "Num devices: " << devCount << std::endl;

  cudaDeviceProp props;
  status = cudaGetDeviceProperties(&props, 0);
  if (status != cudaSuccess)
    throw std::runtime_error("Error getting device properties");
  s << "Device: " << props.name << ": " << props.major << "." << props.minor
    << std::endl;
#endif

  s << "  Global memory:   " << props.totalGlobalMem / mb << " Mb" << std::endl;
  s << "  Shared memory:   " << props.sharedMemPerBlock / kb << " kb"
    << std::endl;
  s << "  Constant memory: " << props.totalConstMem / mb << " Mb" << std::endl;
  s << "  Block registers: " << props.regsPerBlock << std::endl << std::endl;
  s << "  Warp size:         " << props.warpSize << std::endl;
  s << "  Threads per block: " << props.maxThreadsPerBlock << std::endl;
  s << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", "
    << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << " ]"
    << std::endl;
  s << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", "
    << props.maxGridSize[1] << ", " << props.maxGridSize[2] << " ]"
    << std::endl;

  return s.str();
}
//----------------------------------------------------------------------------
#endif

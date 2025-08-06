#include <cuda_runtime.h>
#include <iostream>
int main()
{
  int devCount;
  cudaError_t status = cudaGetDeviceCount(&devCount);
  if (status != cudaSuccess)
    throw std::runtime_error("Error getting device properties");
  std::cout << "Num devices: " << devCount << std::endl;

  cudaDeviceProp props;
  status = cudaGetDeviceProperties(&props, 0);
  if (status != cudaSuccess)
    throw std::runtime_error("Error getting device properties");
  std::cout << "Device: " << props.name << ": " << props.major << "." << props.minor
            << std::endl;
}

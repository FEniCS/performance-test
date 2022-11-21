// Copyright (C) 2021 Igor A. Baratta
// SPDX-License-Identifier:    MIT
#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

namespace CUDA
{
template <class T>
class allocator
{
public:
  using value_type = T;
  using reference = T&;
  using const_reference = const T&;

  allocator() {}

  template <class U>
  allocator(const allocator<U>&)
  {
  }

  /// Allocates memory that will be automatically managed by the Unified Memory
  /// system
  T* allocate(size_t size)
  {
    T* result = nullptr;

    cudaError_t e
        = cudaMallocManaged(&result, size * sizeof(T), cudaMemAttachGlobal);
    std::string error_msg = cudaGetErrorString(e);

    if (e != cudaSuccess)
      throw std::runtime_error("Unable to allocate memory. " + error_msg);

    return result;
  }

  void deallocate(T* ptr, size_t)
  {
    cudaError_t e = cudaFree(ptr);
    std::string error_msg = cudaGetErrorString(e);
    if (e != cudaSuccess)
      throw std::runtime_error("Unable to deallocate memoy" + error_msg);
  }
};

template <class T1, class T2>
bool operator==(const allocator<T1>&, const allocator<T2>&)
{
  return true;
}

template <class T1, class T2>
bool operator!=(const allocator<T1>& lhs, const allocator<T2>& rhs)
{
  return !(lhs == rhs);
}
} // namespace CUDA
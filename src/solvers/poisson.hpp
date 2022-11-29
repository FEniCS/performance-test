#pragma once

#include <cstdint>
#include <cuda_runtime.h>

template <typename T>
void poisson(const std::int32_t N, T* A, const T* w, const T* coordinate_dofs,
             int ndofs_cell, int block_size);
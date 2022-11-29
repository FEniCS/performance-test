#pragma once

#include <cstdint>
#include <cuda_runtime.h>

/// out[indices[i]] += in[i];
template <typename T>
void scatter(std::int32_t N, const std::int32_t* indices, const T* in, T* out,
             int block_size);

/// in[i] = out[indices[i]];
template <typename T>
void gather(std::int32_t N, const std::int32_t* indices, const T* in, T* out,
            int block_size);
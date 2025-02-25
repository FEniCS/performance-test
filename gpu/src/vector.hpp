// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include "util.hpp"

#include <dolfinx/common/Scatterer.h>
#include <dolfinx/common/log.h>
#include <dolfinx/la/dolfinx_la.h>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/transform_reduce.h>

#include <iostream>
#include <type_traits>

namespace
{
template <typename T>
static __global__ void pack(const int N,
                            const std::int32_t* __restrict__ indices,
                            const T* __restrict__ in, T* __restrict__ out)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N)
  {
    out[gid] = in[indices[gid]];
  }
}

template <typename T>
static __global__ void unpack(const int N,
                              const std::int32_t* __restrict__ indices,
                              const T* __restrict__ in, T* __restrict__ out)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N)
  {
    out[indices[gid]] = in[gid];
  }
}

template <typename T>
static __global__ void unpack_add(std::int32_t N,
                                  const int32_t* __restrict__ indices,
                                  const T* __restrict__ in, T* __restrict__ out)
{
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid < N)
  {
    atomicAdd(&out[indices[gid]], in[gid]);
  }
}
} // namespace

namespace dolfinx::acc
{

/// Distributed vector
template <typename T>
class Vector
{
public:
  /// The value type
  using value_type = T;

  /// Create a distributed vector
  Vector(std::shared_ptr<const common::IndexMap> map, int bs)
      : _map(map), _bs(bs),
        _scatterer(std::make_shared<common::Scatterer<>>(*_map, bs))
  {
    int size = bs * (map->size_local() + map->num_ghosts());
    _x = thrust::device_vector<T>(size, 0);

    _buffer_local
        = thrust::device_vector<T>(_scatterer->local_buffer_size(), 0);
    _buffer_remote
        = thrust::device_vector<T>(_scatterer->remote_buffer_size(), 0);
    _local_indices
        = thrust::device_vector<std::int32_t>(_scatterer->local_indices());
    _remote_indices
        = thrust::device_vector<std::int32_t>(_scatterer->remote_indices());

    auto scatterer_type = common::Scatterer<>::type::p2p;
    _request = _scatterer->create_request_vector(scatterer_type);
  }

  // Copy constructor
  Vector(const Vector& x) : _map(x._map), _bs(x._bs), _x(x._x) {}

  // Assignment operator (disabled)
  Vector& operator=(const Vector& x) = delete;

  /// Move Assignment operator
  Vector& operator=(Vector&& x) = default;

  /// Set all entries (including ghosts)
  /// @param[in] v The value to set all entries to (on calling rank)
  void set(T v) { thrust::fill(thrust::device, _x.begin(), _x.end(), v); }

  template <typename OtherVector>
  void copy_from_host(const OtherVector& other)
  {
    // Copies only local data
    thrust::copy(other.array().begin(),
                 other.array().begin() + _map->size_local(), _x.begin());
  }

  template <typename OtherVector>
  void copy(OtherVector& other)
  {
    _x.resize(other.array().size());
    thrust::copy(other.array().begin(), other.array().end(), _x.begin());
  }

  /// Get IndexMap
  std::shared_ptr<const common::IndexMap> map() const { return _map; }

  /// Get block size
  constexpr int bs() const { return _bs; }

  /// Access
  thrust::device_vector<T>& thrust_vector() { return _x; }

  /// Get local part of the vector (const version)
  std::span<const T> array() const
  {
    auto* ptr = thrust::raw_pointer_cast(_x.data());
    return std::span<const T>(ptr, _x.size());
  }

  /// Get local part of the vector
  std::span<T> mutable_array()
  {
    auto* ptr = thrust::raw_pointer_cast(_x.data());
    return std::span<T>(ptr, _x.size());
  }

  void print() const
  {
    // FIXME: printing order is wrong.
    MPI_Comm comm = _map->comm();
    int size, rank;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    for (int i = 0; i < size; i++)
    {
      if (rank == i)
      {
        std::cout << "Rank " << rank << " :" << std::endl;
        thrust::copy(_x.begin(), _x.end(),
                     std::ostream_iterator<value_type>(std::cout, " "));
        std::cout << "\n-----------\n";
      }
      MPI_Barrier(comm);
    }
  }

  /// Begin scatter of local data from owner to ghosts on other ranks
  /// @note Collective MPI operation
  void scatter_fwd_begin(int block_size = 512)
  {
    // TODO: which block_size to use??
    const int num_blocks
        = (_local_indices.size() + block_size - 1) / block_size;
    dim3 dim_block(block_size);
    dim3 dim_grid(num_blocks);

    if (!_local_indices.empty())
    {
      const std::int32_t* indices
          = thrust::raw_pointer_cast(_local_indices.data());
      const T* in = this->array().data();
      T* out = thrust::raw_pointer_cast(_buffer_local.data());
      pack<T><<<dim_grid, dim_block, 0, 0>>>(_local_indices.size(), indices, in,
                                             out);
      device_synchronize();
    }

    T* remote = thrust::raw_pointer_cast(_buffer_remote.data());
    _scatterer->scatter_fwd_begin(
        std::span<const T>(thrust::raw_pointer_cast(_buffer_local.data()),
                           _buffer_local.size()),
        std::span<T>(remote, _buffer_remote.size()),
        std::span<MPI_Request>(_request), common::Scatterer<>::type::p2p);
  }

  void scatter_fwd_end(int block_size = 512)
  {
    spdlog::debug("scatter_fwd_end start");
    // TODO: which block_size to use??
    const std::int32_t local_size = _bs * _map->size_local();
    const std::int32_t num_ghosts = _bs * _map->num_ghosts();
    _scatterer->scatter_fwd_end(std::span<MPI_Request>(_request));

    spdlog::debug("scatter_fwd_end step 1");

    spdlog::debug("scatter_fwd_end local buf size = {}, remote buf size {}",
                  _buffer_local.size(), _buffer_remote.size());

    const int num_blocks
        = (_remote_indices.size() + block_size - 1) / block_size;
    dim3 dim_block(block_size);
    dim3 dim_grid(num_blocks);
    std::span<T> x_remote(this->mutable_array().data() + local_size,
                          num_ghosts);

    spdlog::debug("scatter_fwd_end step 2");

    if (!_remote_indices.empty())
    {
      const std::int32_t* indices
          = thrust::raw_pointer_cast(_remote_indices.data());
      const T* in = thrust::raw_pointer_cast(_buffer_remote.data());
      T* out = x_remote.data();
      unpack<T><<<dim_grid, dim_block, 0, 0>>>(_remote_indices.size(), indices,
                                               in, out);
      device_synchronize();
    }
    spdlog::debug("scatter_fwd_end end");
  }

  /// Scatter local data to ghost positions on other ranks
  /// @note Collective MPI operation
  void scatter_fwd()
  {
    this->scatter_fwd_begin();
    this->scatter_fwd_end();
  }

  // Pack data, start reverse scatter
  void scatter_rev_begin(int block_size = 512)
  {
    // TODO: which block_size to use??
    const int num_blocks
        = (_remote_indices.size() + block_size - 1) / block_size;
    dim3 dim_block(block_size);
    dim3 dim_grid(num_blocks);

    const std::int32_t local_size = _bs * _map->size_local();
    const std::int32_t* indices
        = thrust::raw_pointer_cast(_remote_indices.data());
    const T* in = this->array().data() + local_size;
    T* out = thrust::raw_pointer_cast(_buffer_remote.data());
    pack<T><<<dim_grid, dim_block, 0, 0>>>(_remote_indices.size(), indices, in,
                                           out);
    device_synchronize();

    T* local = thrust::raw_pointer_cast(_buffer_local.data());
    _scatterer->scatter_rev_begin(
        std::span<const T>(out, _buffer_remote.size()),
        std::span<T>(local, _buffer_local.size()),
        std::span<MPI_Request>(_request), common::Scatterer<>::type::p2p);
  }

  // Finalize reverse scatter, unpack data
  void scatter_rev_end(int block_size = 512)
  {
    // TODO: which block_size to use??
    const std::int32_t local_size = _bs * _map->size_local();
    _scatterer->scatter_rev_end(std::span<MPI_Request>(_request));

    const int num_blocks
        = (_local_indices.size() + block_size - 1) / block_size;
    dim3 dim_block(block_size);
    dim3 dim_grid(num_blocks);
    std::span<T> x_local(this->mutable_array().data(), local_size);

    const std::int32_t* indices
        = thrust::raw_pointer_cast(_local_indices.data());
    const T* in = thrust::raw_pointer_cast(_buffer_local.data());
    T* out = x_local.data();
    unpack_add<T><<<dim_grid, dim_block, 0, 0>>>(_local_indices.size(), indices,
                                                 in, out);
    device_synchronize();
  }

  /// Scatter local data from ghosts, and accumulate in owned part of vector
  /// @note Collective MPI operation
  void scatter_rev()
  {
    this->scatter_rev_begin();
    this->scatter_rev_end();
  }

  /// Copy data back to host
  std::vector<T> data_copy()
  {
    std::vector<T> data(_x.size());
    thrust::copy(_x.begin(), _x.end(), data.begin());
    return data;
  }

private:
  // Map describing the data layout
  std::shared_ptr<const common::IndexMap> _map;

  // Block size
  int _bs;

  // Scatter for managing MPI communication
  std::shared_ptr<common::Scatterer<>> _scatterer;

  // MPI request handle
  std::vector<MPI_Request> _request = {MPI_REQUEST_NULL};

  // Buffers for ghost scatters
  thrust::device_vector<T> _buffer_local, _buffer_remote;

  // indices for ghost scatters
  thrust::device_vector<std::int32_t> _local_indices, _remote_indices;

  // Vector data
  thrust::device_vector<T> _x;
};

/// Compute the inner product of two vectors. The two vectors must have
/// the same parallel layout
/// @note Collective MPI operation
/// @param a A vector
/// @param b A vector
/// @return Returns `a^{H} b` (`a^{T} b` if `a` and `b` are real)
template <typename Vector>
auto inner_product(const Vector& a, const Vector& b)
{
  using T = typename Vector::value_type;

  const std::int32_t local_size = a.bs() * a.map()->size_local();
  std::span<const T> x_a = a.array().subspan(0, local_size);
  std::span<const T> x_b = b.array().subspan(0, local_size);

  if (local_size != b.bs() * b.map()->size_local())
    throw std::runtime_error("Incompatible vector sizes");

  T local = thrust::inner_product(thrust::device, x_a.begin(), x_a.end(),
                                  x_b.begin(), T{0.0});

  T result;
  MPI_Allreduce(&local, &result, 1, dolfinx::MPI::mpi_t<T>, MPI_SUM,
                a.map()->comm());
  return result;
}

/// Compute the squared L2 norm of vector
/// @note Collective MPI operation
template <typename Vector>
auto squared_norm(const Vector& a)
{
  using T = typename Vector::value_type;
  T result = inner_product(a, a);
  return std::real(result);
}

/// Compute the norm of the vector
/// @note Collective MPI operation
/// @param a A vector
/// @param type Norm type (supported types are \f$L^2\f$ and \f$L^\infty\f$)
template <typename Vector>
auto norm(const Vector& a, dolfinx::la::Norm type = dolfinx::la::Norm::l2)
{
  using T = typename Vector::value_type;

  switch (type)
  {
  case dolfinx::la::Norm::l2:
    return std::sqrt(squared_norm(a));
  case dolfinx::la::Norm::linf:
  {
    const std::int32_t size_local = a.bs() * a.map()->size_local();
    std::span<const T> x_a = a.array().subspan(0, size_local);
    auto max_pos = thrust::max_element(thrust::device, x_a.begin(), x_a.end());
    auto local_linf = std::abs(*max_pos);
    decltype(local_linf) linf = 0;
    MPI_Allreduce(&local_linf, &linf, 1, dolfinx::MPI::mpi_t<decltype(linf)>,
                  MPI_MAX, a.map()->comm());
    return linf;
  }
  default:
    throw std::runtime_error("Norm type not supported");
  }
}

/// Compute vector r = alpha*x + y
/// @param r Result
/// @param alpha
/// @param x
/// @param y
template <typename Vector, typename S>
void axpy(Vector& r, S alpha, const Vector& x, const Vector& y)
{
  spdlog::debug("AXPY start");
  using T = typename Vector::value_type;
  thrust::transform(thrust::device, x.array().begin(),
                    x.array().begin() + x.map()->size_local(),
                    y.array().begin(), r.mutable_array().begin(),
                    [alpha] __host__ __device__(const T& vx, const T& vy)
                    { return vx * alpha + vy; });
  spdlog::debug("AXPY end");
}

/// Scale vector by alpha
/// @param r Result
/// @param alpha
template <typename Vector, typename S>
void scale(Vector& r, S alpha)
{
  using T = typename Vector::value_type;
  thrust::for_each(thrust::device, r.mutable_array().begin(),
                   r.mutable_array().end(),
                   [alpha] __host__ __device__(T & v) { v *= alpha; });
}

/// Compute vector a = b
/// @param a
/// @param b
template <typename Vector>
void copy(Vector& a, const Vector& b)
{
  using T = typename Vector::value_type;
  const std::int32_t local_size = a.bs() * a.map()->size_local();
  std::span<T> x_a = a.mutable_array().subspan(0, local_size);
  std::span<const T> x_b = b.array().subspan(0, local_size);
  thrust::copy(thrust::device, x_b.begin(), x_b.end(), x_a.begin());
}

/// Compute pointwise vector multiplication w[i] = x[i] * y[i]
/// @param w
/// @param x
/// @param y
template <typename Vector>
void pointwise_mult(Vector& w, const Vector& x, const Vector& y)
{
  spdlog::debug("pointwise_mult start");

  using T = typename Vector::value_type;
  thrust::transform(thrust::device, x.array().begin(),
                    x.array().begin() + x.map()->size_local(),
                    y.array().begin(), w.mutable_array().begin(),
                    [] __host__ __device__(const T& xi, const T& yi)
                    { return xi * yi; });
  spdlog::debug("pointwise_mult end");
}

template <typename Vector, typename UnaryFunction>
void transform(Vector& x, UnaryFunction op)
{
  thrust::transform(thrust::device, x.array().begin(),
                    x.array().begin() + x.map()->size_local(),
                    x.mutable_array().begin(), op);
}

} // namespace dolfinx::acc

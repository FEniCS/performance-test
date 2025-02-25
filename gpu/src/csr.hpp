#pragma once

#include <dolfinx.h>
#include <dolfinx/fem/dolfinx_fem.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/la/MatrixCSR.h>

#include <thrust/device_vector.h>

#include "util.hpp"

namespace
{
// // /// Computes y += A*x for a local CSR matrix A and local dense vectors x,y
/// @param[in] values Nonzero values of A
/// @param[in] row_begin First index of each row in the arrays values and
/// indices.
/// @param[in] row_end Last index of each row in the arrays values and indices.
/// @param[in] indices Column indices for each non-zero element of the matrix A
/// @param[in] x Input vector
/// @param[in, out] y Output vector
template <typename T>
__global__ void spmv_impl(int N, const T* values, const std::int32_t* row_begin,
                          const std::int32_t* row_end, const std::int32_t* indices, const T* x,
                          T* y)
{
  // Calculate the row index for this thread.
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Check if the row index is out of bounds.
  if (i < N)
  {
    // Perform the sparse matrix-vector multiplication for this row.
    T vi{0};
    for (std::int32_t j = row_begin[i]; j < row_end[i]; j++)
      vi += values[j] * x[indices[j]];
    y[i] += vi;
  }
}

template <typename T>
__global__ void spmvT_impl(int N, const T* values, const std::int32_t* row_begin,
                           const std::int32_t* row_end, const std::int32_t* indices, const T* x,
                           T* y)
{
  // Calculate the row index for this thread.
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // Check if the row index is out of bounds.
  if (i < N)
  {
    // Perform the transpose sparse matrix-vector multiplication for this row.
    for (std::int32_t j = row_begin[i]; j < row_end[i]; j++)
      atomicAdd(&y[indices[j]], values[j] * x[i]);
  }
}

} // namespace

namespace dolfinx::acc
{
template <typename T>
class MatrixOperator
{
public:
  /// The value type
  using value_type = T;

  /// Create a distributed vector
  MatrixOperator(std::shared_ptr<fem::Form<T, T>> a,
                 std::vector<std::reference_wrapper<const fem::DirichletBC<T>>> bcs)
  {

    dolfinx::common::Timer t0("~setup phase MatrixOperator");

    if (a->rank() != 2)
      throw std::runtime_error("Form should have rank be 2.");

    auto V = a->function_spaces()[0];
    la::SparsityPattern pattern = fem::create_sparsity_pattern(*a);
    pattern.finalize();
    _col_map = std::make_shared<const common::IndexMap>(pattern.column_index_map());
    _row_map = V->dofmap()->index_map;

    _A = std::make_unique<
        la::MatrixCSR<T, std::vector<T>, std::vector<std::int32_t>, std::vector<std::int32_t>>>(
        pattern);
    fem::assemble_matrix(_A->mat_add_values(), *a, bcs);
    _A->scatter_rev();
    fem::set_diagonal<T>(_A->mat_set_values(), *V, bcs, T(1.0));

    // Get communicator from mesh
    _comm = V->mesh()->comm();

    std::int32_t num_rows = _row_map->size_local();
    std::int32_t nnz = _A->row_ptr()[num_rows];
    _nnz = nnz;

    T norm = 0.0;
    for (T v : _A->values())
      norm += v * v;

    spdlog::info("A norm = {}", std::sqrt(norm));

    // Get inverse diagonal entries (for Jacobi preconditioning)
    std::vector<T> diag_inv(num_rows);
    for (int i = 0; i < num_rows; ++i)
    {
      for (int j = _A->row_ptr()[i]; j < _A->row_ptr()[i + 1]; ++j)
      {
        if (_A->cols()[j] == i)
          diag_inv[i] = 1.0 / _A->values()[j];
      }
    }
    _diag_inv = thrust::device_vector<T>(diag_inv.size());
    thrust::copy(diag_inv.begin(), diag_inv.end(), _diag_inv.begin());

    _row_ptr = thrust::device_vector<std::int32_t>(num_rows + 1);
    _off_diag_offset = thrust::device_vector<std::int32_t>(num_rows);
    _cols = thrust::device_vector<std::int32_t>(nnz);
    _values = thrust::device_vector<T>(nnz);

    // Copy data from host to device
    spdlog::warn("Creating Device matrix with {} non zeros", _nnz);
    spdlog::warn("Creating row_ptr with {} to {}", num_rows + 1, _row_ptr.size());
    thrust::copy(_A->row_ptr().begin(), _A->row_ptr().begin() + num_rows + 1, _row_ptr.begin());
    spdlog::warn("Creating off_diag with {} to {}", _A->off_diag_offset().size(),
                 _off_diag_offset.size());
    thrust::copy(_A->off_diag_offset().begin(), _A->off_diag_offset().begin() + num_rows,
                 _off_diag_offset.begin());
    spdlog::warn("Creating cols with {} to {}", nnz, _cols.size());
    thrust::copy(_A->cols().begin(), _A->cols().begin() + nnz, _cols.begin());
    spdlog::warn("Creating values with {} to {}", nnz, _values.size());
    thrust::copy(_A->values().begin(), _A->values().begin() + nnz, _values.begin());
  }

  MatrixOperator(const fem::FunctionSpace<T>& V0, const fem::FunctionSpace<T>& V1)
  {
    dolfinx::common::Timer t0("~setup phase Interpolation Operators");
    _comm = V0.mesh()->comm();
    assert(V0.mesh());
    auto mesh = V0.mesh();
    assert(V1.mesh());
    assert(mesh == V1.mesh());

    std::shared_ptr<const fem::DofMap> dofmap0 = V0.dofmap();
    assert(dofmap0);
    std::shared_ptr<const fem::DofMap> dofmap1 = V1.dofmap();
    assert(dofmap1);

    // Create and build  sparsity pattern
    assert(dofmap0->index_map);
    assert(dofmap1->index_map);

    la::SparsityPattern pattern(_comm, {dofmap1->index_map, dofmap0->index_map},
                                {dofmap1->index_map_bs(), dofmap0->index_map_bs()});

    int tdim = mesh->topology()->dim();
    auto map = mesh->topology()->index_map(tdim);
    assert(map);
    std::vector<std::int32_t> c(map->size_local(), 0);
    std::iota(c.begin(), c.end(), 0);
    fem::sparsitybuild::cells(pattern, {c, c}, {*dofmap1, *dofmap0});
    pattern.finalize();

    // Build operator
    _A = std::make_unique<
        la::MatrixCSR<T, std::vector<T>, std::vector<std::int32_t>, std::vector<std::int32_t>>>(
        pattern);

    // FIXME: should this be mat_add or mat_set?
    fem::interpolation_matrix<T>(V0, V1, _A->mat_set_values());

    // Create HIP/CUDA matrix
    _col_map = std::make_shared<const common::IndexMap>(pattern.column_index_map());
    _row_map = V1.dofmap()->index_map;
    std::int32_t num_rows = _row_map->size_local();
    std::int32_t nnz = _A->row_ptr()[num_rows];
    _nnz = nnz;

    spdlog::warn("Operator Number of non zeros {}", _nnz);
    spdlog::warn("Operator Number of rows {}", num_rows);
    spdlog::warn("Operator dm0 size {}", V0.dofmap()->index_map->size_global());
    spdlog::warn("Operator dm1 size {}", V1.dofmap()->index_map->size_global());
    spdlog::warn("Max column = {}", *std::max_element(_A->cols().begin(), _A->cols().end()));

    T norm = 0.0;
    auto v = _A->values();
    for (int i = 0; i < nnz; ++i)
      norm += v[i] * v[i];

    double global_norm = 0;
    MPI_Allreduce(&norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    spdlog::info("A interp norm = {}", std::sqrt(global_norm));

    _row_ptr = thrust::device_vector<std::int32_t>(num_rows + 1);
    _off_diag_offset = thrust::device_vector<std::int32_t>(num_rows);
    _cols = thrust::device_vector<std::int32_t>(nnz);
    _values = thrust::device_vector<T>(nnz);

    // Copy data from host to device
    thrust::copy(_A->row_ptr().begin(), _A->row_ptr().begin() + num_rows + 1, _row_ptr.begin());
    thrust::copy(_A->off_diag_offset().begin(), _A->off_diag_offset().begin() + num_rows,
                 _off_diag_offset.begin());
    thrust::copy(_A->cols().begin(), _A->cols().begin() + nnz, _cols.begin());
    thrust::copy(_A->values().begin(), _A->values().begin() + nnz, _values.begin());
  }

  template <typename Vector>
  void get_diag_inverse(Vector& diag_inv)
  {
    thrust::copy(_diag_inv.begin(), _diag_inv.end(), diag_inv.mutable_array().begin());
  }

  /**
   * @brief The matrix-vector multiplication operator, which multiplies the
   * matrix with the input vector and stores the result in the output vector.
   *
   * @tparam Vector  The type of the input and output vector.
   *
   * @param x        The input vector.
   * @param y        The output vector.
   */
  template <typename Vector>
  void operator()(Vector& x, Vector& y, bool transpose = false)
  {
    dolfinx::common::Timer t0("% MatrixOperator application");

    y.set(T{0});
    T* _x = x.mutable_array().data();
    T* _y = y.mutable_array().data();

    if (transpose)
    {
      int num_rows = _row_map->size_local();
      dim3 block_size(256);
      dim3 grid_size((num_rows + block_size.x - 1) / block_size.x);
      x.scatter_fwd_begin();
      spmvT_impl<T>
          <<<grid_size, block_size, 0, 0>>>(num_rows, thrust::raw_pointer_cast(_values.data()),
                                            thrust::raw_pointer_cast(_row_ptr.data()),
                                            thrust::raw_pointer_cast(_off_diag_offset.data()),
                                            thrust::raw_pointer_cast(_cols.data()), _x, _y);
      check_device_last_error();
      x.scatter_fwd_end();

      spmvT_impl<T>
          <<<grid_size, block_size, 0, 0>>>(num_rows, thrust::raw_pointer_cast(_values.data()),
                                            thrust::raw_pointer_cast(_off_diag_offset.data()),
                                            thrust::raw_pointer_cast(_row_ptr.data()) + 1,
                                            thrust::raw_pointer_cast(_cols.data()), _x, _y);
      check_device_last_error();
    }
    else
    {
      int num_rows = _row_map->size_local();
      dim3 block_size(256);
      dim3 grid_size((num_rows + block_size.x - 1) / block_size.x);
      x.scatter_fwd_begin();
      spmv_impl<T>
          <<<grid_size, block_size, 0, 0>>>(num_rows, thrust::raw_pointer_cast(_values.data()),
                                            thrust::raw_pointer_cast(_row_ptr.data()),
                                            thrust::raw_pointer_cast(_off_diag_offset.data()),
                                            thrust::raw_pointer_cast(_cols.data()), _x, _y);
      check_device_last_error();
      x.scatter_fwd_end();

      spmv_impl<T>
          <<<grid_size, block_size, 0, 0>>>(num_rows, thrust::raw_pointer_cast(_values.data()),
                                            thrust::raw_pointer_cast(_off_diag_offset.data()),
                                            thrust::raw_pointer_cast(_row_ptr.data()) + 1,
                                            thrust::raw_pointer_cast(_cols.data()), _x, _y);
      check_device_last_error();
    }

    device_synchronize();
  }

  std::shared_ptr<const common::IndexMap> column_index_map() { return _col_map; }

  std::shared_ptr<const common::IndexMap> row_index_map() { return _row_map; }

  std::size_t nnz() { return _nnz; }

  ~MatrixOperator() {}

private:
  std::size_t _nnz;
  thrust::device_vector<T> _values;
  thrust::device_vector<T> _diag_inv;
  thrust::device_vector<std::int32_t> _row_ptr;
  thrust::device_vector<std::int32_t> _cols;
  thrust::device_vector<std::int32_t> _off_diag_offset;
  std::shared_ptr<const common::IndexMap> _col_map, _row_map;
  std::unique_ptr<
      la::MatrixCSR<T, std::vector<T>, std::vector<std::int32_t>, std::vector<std::int32_t>>>
      _A;

  MPI_Comm _comm;
};
} // namespace dolfinx::acc

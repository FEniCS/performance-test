// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#pragma once

#include "util.hpp"
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>

#include <basix/finite-element.h>
#include <basix/interpolation.h>
#include <basix/quadrature.h>
#include <iomanip>

template <typename T, int P, int Q>
__constant__ T phi0_const[Q * (P + 1)];

template <typename T, int Q>
__constant__ T dphi1_const[Q * Q];

/// @brief Compute 3d index from 1d indices.
///
/// Compute the index `idx = ld0 * i + ld1 * j + ld2 * k`.
///
/// For contiguous, row-major storage of a tensor with shape `(n0, n1,
/// n2)`, use `ld0=n1*n2`, `ld1=n2`, `ld2=1` (`k` varies fastest,
/// followed by `j`).
///
/// For contiguous, column-major storage of a tensor with shape `(n0,
/// n1, n2)`, use `ld0=1`, `ld1=n0`, `ld2=n0*n1` (`i` varies fastest,
/// followed by `j`).
///
/// For contiguous storage with `j` varying fastest and `i` slowest, use
/// `ld0=n1*n2`, `ld1=1`, `ld2=n1`
///
/// For contiguous storage with `j` varying fastest and `k` slowest, use
/// `ld0=n1`, `ld1=1`, `ld2=n0*n1`
///
/// @tparam ld0 Stride for first (`i`) index.
/// @tparam ld1 Stride for second (`k`) index.
/// @tparam ld2 Stride for third (`k`) index.
/// @param[in] i
/// @param[in] j
/// @param[in] k
/// @return Flattened index.
template <int ld0, int ld1, int ld2>
__device__ __forceinline__ int ijk(int i, int j, int k)
{
  return i * ld0 + j * ld1 + k * ld2;
}

/// @brief TODO
/// @tparam T
/// @param mat
/// @param shape
/// @return
template <typename T>
bool matrix_is_identity(const std::vector<T>& mat,
                        std::array<std::size_t, 2> shape)
{
  T eps = std::numeric_limits<T>::epsilon();
  if (shape[0] == shape[1])
  {
    for (std::size_t i = 0; i < shape[0]; ++i)
    {
      for (std::size_t j = 0; j < shape[1]; ++j)
      {
        if (i != j and std::abs(mat[i * shape[1] + j]) > 5 * eps)
          return false;
        else if (i == j and std::abs(mat[i * shape[1] + j] - 1.0) > 5 * eps)
          return false;
      }
    }

    return true;
  }
  else
    return false;
}

/// @brief Computes weighted geometry tensor G from the coordinates and
/// quadrature weights.
/// @param [in] xgeom Geometry points [*, 3]
/// @param [out] G_entity geometry data [n_entities, nq, 6]
/// @param [in] geometry_dofmap Location of coordinates for each cell in
/// xgeom [*, ncdofs]
/// @param [in] _dphi Basis derivative tabulation for cell at quadrature
/// points [3, nq, ncdofs]
/// @param [in] weights Quadrature weights [nq]
/// @param [in] entities list of cells to compute for [n_entities]
/// @param [in] n_entities total number of cells to compute for
/// @tparam T scalar type
/// @tparam Q number of quadrature points (in 1D)
template <typename T, int Q>
__global__ void geometry_computation(const T* xgeom, T* G_entity,
                                     const std::int32_t* geometry_dofmap,
                                     const T* _dphi, const T* weights,
                                     const int* entities, int n_entities)
{
  // One block per cell
  int c = blockIdx.x;

  // Limit to cells in list
  if (c >= n_entities)
    return;

  // Cell index
  int cell = entities[c];

  // Number of quadrature points (must match arrays in weights and dphi)
  constexpr int nq = Q * Q * Q;

  // Number of coordinate dofs
  constexpr int ncdofs = 8;

  // Geometric dimension
  constexpr int gdim = 3;

  extern __shared__ T shared_mem[];

  // coord_dofs has shape [ncdofs, gdim]
  T* _coord_dofs = shared_mem;

  // First collect geometry into shared memory
  int iq = threadIdx.x;
  if constexpr (nq < 27)
  {
    // Only 8 threads when Q == 2
    assert(iq < 8);
    for (int j = 0; j < 3; ++j)
      _coord_dofs[iq * 3 + j]
          = xgeom[3 * geometry_dofmap[cell * ncdofs + iq] + j];
  }
  else
  {
    int i = iq / gdim;
    int j = iq % gdim;
    if (i < ncdofs)
      _coord_dofs[iq] = xgeom[3 * geometry_dofmap[cell * ncdofs + i] + j];
  }

  __syncthreads();
  // One quadrature point per thread

  if (iq >= nq)
    return;

  // Jacobian
  T J[3][3];
  auto coord_dofs = [&_coord_dofs](int i, int j) -> T&
  { return _coord_dofs[i * gdim + j]; };

  // For each quadrature point / thread
  {
    // dphi has shape [gdim, ncdofs]
    auto dphi = [&_dphi, iq](int i, int j) -> const T
    { return _dphi[(i * nq + iq) * ncdofs + j]; };
    for (std::size_t i = 0; i < gdim; i++)
    {
      for (std::size_t j = 0; j < gdim; j++)
      {
        J[i][j] = 0.0;
        for (std::size_t k = 0; k < ncdofs; k++)
          J[i][j] += coord_dofs(k, i) * dphi(j, k);
      }
    }

    // Components of K = J^-1 (detJ)
    T K[3][3] = {{J[1][1] * J[2][2] - J[1][2] * J[2][1],
                  -J[0][1] * J[2][2] + J[0][2] * J[2][1],
                  J[0][1] * J[1][2] - J[0][2] * J[1][1]},
                 {-J[1][0] * J[2][2] + J[1][2] * J[2][0],
                  J[0][0] * J[2][2] - J[0][2] * J[2][0],
                  -J[0][0] * J[1][2] + J[0][2] * J[1][0]},
                 {J[1][0] * J[2][1] - J[1][1] * J[2][0],
                  -J[0][0] * J[2][1] + J[0][1] * J[2][0],
                  J[0][0] * J[1][1] - J[0][1] * J[1][0]}};

    T detJ = J[0][0] * K[0][0] - J[0][1] * K[1][0] + J[0][2] * K[2][0];

    int offset = (c * nq * 6 + iq);
    G_entity[offset]
        = (K[0][0] * K[0][0] + K[0][1] * K[0][1] + K[0][2] * K[0][2])
          * weights[iq] / detJ;
    G_entity[offset + nq]
        = (K[1][0] * K[0][0] + K[1][1] * K[0][1] + K[1][2] * K[0][2])
          * weights[iq] / detJ;
    G_entity[offset + 2 * nq]
        = (K[2][0] * K[0][0] + K[2][1] * K[0][1] + K[2][2] * K[0][2])
          * weights[iq] / detJ;
    G_entity[offset + 3 * nq]
        = (K[1][0] * K[1][0] + K[1][1] * K[1][1] + K[1][2] * K[1][2])
          * weights[iq] / detJ;
    G_entity[offset + 4 * nq]
        = (K[2][0] * K[1][0] + K[2][1] * K[1][1] + K[2][2] * K[1][2])
          * weights[iq] / detJ;
    G_entity[offset + 5 * nq]
        = (K[2][0] * K[2][0] + K[2][1] * K[2][1] + K[2][2] * K[2][2])
          * weights[iq] / detJ;
  }
}

/// Compute b = A * u where A is the stiffness operator for a set of
/// entities (cells or facets) in a mesh.
///
/// The stiffness operator is defined as:
///
///     A = ∑_i ∫_Ω C ∇ϕ_i ∇ϕ_j dx
///
/// where C is a constant, ϕ_i and ϕ_j are the basis functions of the
/// finite element space, and ∇ϕ_i is the gradient of the basis
/// function. The integral is computed over the domain Ω of the entity
/// using sum factorization. The basis functions are defined on the
/// reference element and are transformed to the physical element using
/// the geometry operator G. G is a 3x3 matrix per quadrature point per
/// entity.
///
/// @tparam T Data type of the input and output arrays
/// @tparam P Polynomial degree of the basis functions
/// @tparam Q Number of quadrature points in 1D
/// @param u Input vector of size (ndofs,)
/// @param entity_constants Array of size (n_entities,) with the
/// constant C for each entity
/// @param b Output vector of size (ndofs,)
/// @param G_entity Array of size (n_entities, nq, 6) with the geometry
/// operator G for each entity
/// @param entity_dofmap Array of size (n_entities, ndofs) with the
/// dofmap for each entity
/// @param phi0_in Array of size (nq, ndofs) with the interpolation basis
/// functions in 1D. u1_i = phi0_(ij) u_j, where u are the dofs
/// associated with the element (degree P), and u1 are the dofs for the
/// finite elment (degree >= P) that u is interpolated into.
/// @param dphi1_in Array of size (nq, nq) with the 1D basis function
/// derivatives. FIXME: layout is (point_idx, dphi_i)?
/// @param entities List of entities to compute on
/// @param n_entities Number of entries in `entities`
/// @param bc_marker Array of size (ndofs,) with the boundary condition
/// marker
/// @param identity If 1, the basis functions are the identity for the
/// given quadrature points
///
/// @note The kernel is launched with a 3D grid of 1D blocks, where each
/// block is responsible for computing the stiffness operator for a
/// single entity. The block size is (P+1, P+1, P+1) and the shared
/// memory 2 * (P+1)^3 * sizeof(T).
template <typename T, int P, int Q>
__launch_bounds__(Q* Q* Q) __global__
    void stiffness_operator(const T* __restrict__ u,
                            const T* __restrict__ entity_constants,
                            T* __restrict__ b, const T* __restrict__ G_entity,
                            const std::int32_t* __restrict__ entity_dofmap,
                            const int* __restrict__ entities, int n_entities,
                            const std::int8_t* __restrict__ bc_marker,
                            bool identity)
{
  // Note: each thread is respinsible for one quadrature point. Since
  // the number of DOFs is less than or equal to the number quadrature
  // points a subset of threads are also responsible for one DOF
  // contribiution (an entry in b).

  constexpr int nd = P + 1; // Number of dofs per direction in 1D
  constexpr int nq = Q;     // Number of quadrature points in 1D

  assert(blockDim.x == nq);
  assert(blockDim.y == nq);
  assert(blockDim.z == nq);

  // block_id is the cell (or facet) index
  const int block_id = blockIdx.x;

  // Check if the block_id is valid (i.e. within the number of entities)
  if (block_id >= n_entities) // Should always be true
    return;

  constexpr int square_nd = nd * nd;
  constexpr int square_nq = nq * nq;
  constexpr int cube_nd = square_nd * nd;
  constexpr int cube_nq = square_nq * nq;

  constexpr int nq1 = nq + 1;

  // Try padding
  __shared__ T scratch1[cube_nq];
  __shared__ T scratch2[cube_nq];
  __shared__ T scratch3[cube_nq];

  // Note: thread order on the device is such that
  // neighboring threads can get coalesced memory access, i.e.
  // tz threads are closest together (called .x by CUDA/HIP)
  const int tx = threadIdx.z; // 1d dofs x direction
  const int ty = threadIdx.y; // 1d dofs y direction
  const int tz = threadIdx.x; // 1d dofs z direction

  // thread_id represents the quadrature index in 3D ('row-major')
  const int thread_id = tx * square_nq + ty * nq + tz;

  // Copy phi and dphi to shared memory
  __shared__ T phi0[nq * nd];
  __shared__ T dphi1[nq1 * nq];

  if (thread_id < nd * nq)
    phi0[thread_id] = phi0_const<T, P, Q>[thread_id];

  if (tz < nq and ty < nq and tx == 0)
    dphi1[ty * nq1 + tz] = dphi1_const<T, Q>[ty * nq + tz];

  // Get dof value (in x) that this thread is responsible for, and
  // place in shared memory.
  int dof = -1;

  // Note: We might have more threads per block than dofs, so we need
  // to check if the thread_id is valid
  scratch2[ijk<square_nq, nq, 1>(tx, ty, tz)] = 0;
  if (tx < nd && ty < nd && tz < nd)
  {
    int dof_thread_id = tx * square_nd + ty * nd + tz;
    int entity_index = entities[block_id];
    dof = entity_dofmap[entity_index * cube_nd + dof_thread_id];
    if (bc_marker[dof])
    {
      b[dof] = u[dof];
      dof = -1;
    }
    else
      scratch2[ijk<square_nq, nq, 1>(tx, ty, tz)] = u[dof];
  }

  __syncthreads(); // Make sure all threads have written to shared memory

  // Interpolate basis functions to quadrature points
  if (identity != 1)
  {
    // Interpolate u from phi0 basis to phi1 basis. The phi1 basis nodes
    // are collcated with the quadrature points.
    //
    // Note: phi0 has shape (nq, nd)
    //
    // u(q0, q1, q2) = \phi0_(i)(q0) \phi0_(j)(q1) \phi0_(k)(q2) u_(i, j, k)
    //
    // 0. tmp0_(q0, j, k)  = \phi0_(i)(q0) u_(i, j, k)
    // 1. tmp1_(q0, q1, k) = \phi0_(j)(q1) tmp0_(q0, j, k)
    // 2. u_(q0, q1, q2)   = \phi0_(k)(q2) tmp1_(q0, q1, k)

    // 0. tmp0_(q0, j, k) = \phi_(i)(q0) u_(i, j, k)
    T xq = 0;
    for (int ix = 0; ix < nd; ++ix)
      xq += phi0[tx * nd + ix] * scratch2[ijk<square_nq, nq, 1>(ix, ty, tz)];

    scratch1[ijk<square_nq, nq, 1>(tx, ty, tz)] = xq;
    __syncthreads();

    // 1. tmp1_(q0, q1, k) = \phi0_(j)(q1) tmp0_(q0, j, k)
    xq = 0;
    for (int iy = 0; iy < nd; ++iy)
      xq += phi0[ty * nd + iy] * scratch1[ijk<square_nq, nq, 1>(tx, iy, tz)];

    scratch3[ijk<square_nq, nq, 1>(tx, ty, tz)] = xq;
    __syncthreads();

    // 2. u_(q0, q1, q2) = \phi_(k)(q2) tmp1_(q0, q1, k)
    xq = 0;
    for (int iz = 0; iz < nd; ++iz)
      xq += phi0[tz * nd + iz] * scratch3[ijk<square_nq, nq, 1>(tx, ty, iz)];

    scratch2[ijk<square_nq, nq, 1>(tx, ty, tz)] = xq;
    __syncthreads();
  } // end of interpolation

  // Compute du/dx0, du/dx1 and du/dx2 (deriavtives on reference cell)
  // at the quadrature point computed by this thread.
  //
  // From
  //
  //   u(q0, q1, q2) = \phi0_(i)(q0) \phi0_(j)(q1) \phi0_(k)(q2) u_(i, j, k)
  //
  // we have
  //
  //   (du/dx0)(q0, q1, q2) = (d\phi1_(i)/dx)(q0) \phi1_(j)(q1) \phi1_(k)(q2)
  //   u_(i, j, k) (du/dx1)(q0, q1, q2) = \phi1_(i)(q0) (d\phi1_(j)/dx)(q1)
  //   \phi1_(k)(q2) u_(i, j, k) (du/dx2)(q0, q1, q2) = \phi1_(i)(q0)
  //   \phi1_(j)(q1) (d\phi1_(k)/dx)(q1) u_(i, j, k)
  //
  // Quadrature points and the phi1 'degrees-of-freedom' are co-located,
  // therefore:
  //
  //   (du/dx0)(q0, q1, q2)
  //    = (d\phi1_(i)/dx)(q0) \phi1_(j)(q1) \phi1_(k)(q2) u_(i, j, k)
  //    = (d\phi1_(i)/dx)(q0) \delta_(q1, j) \delta_(q2, k) u_(i, j, k)
  //    = (d\phi1_i)/dx)_(q0) u_(i, q1, q2)
  //
  //   (du/dx1)(q0, q1, q2) = (d\phi1_(j)/dx)(q1) u_(q0, j, q2)
  //   (du/dx2)(q0, q1, q2) = (d\phi1_(j)/dx)(q2) u_(q1, q1, k)

  // (du/dx0)_(q0, q1, q2) = (d\phi1_(i)/dx)(q0) u_(i, q1, q2)
  T val_x = 0;
  for (int ix = 0; ix < nq; ++ix)
    val_x += dphi1[tx * nq1 + ix] * scratch2[ijk<square_nq, nq, 1>(ix, ty, tz)];

  // (du/dx1)_(q0, q1, q2) = (d\phi1_(j)/dx)(q1) u_(q0, j, q2)
  T val_y = 0;
  for (int iy = 0; iy < nq; ++iy)
    val_y += dphi1[ty * nq1 + iy] * scratch2[ijk<square_nq, nq, 1>(tx, iy, tz)];

  // (du/dx2)_(q0, q1, q2) = (d\phi1_(k)/dx)(q2) u_(q0, q1, k)
  T val_z = 0;
  for (int iz = 0; iz < nq; ++iz)
    val_z += dphi1[tz * nq1 + iz] * scratch2[ijk<square_nq, nq, 1>(tx, ty, iz)];

  // TODO: Add some maths

  // Apply geometric transformation to data at quadrature point
  const int gid = block_id * cube_nq * 6 + thread_id;
  const T G0 = non_temp_load(&G_entity[gid + cube_nq * 0]);
  const T G1 = non_temp_load(&G_entity[gid + cube_nq * 1]);
  const T G2 = non_temp_load(&G_entity[gid + cube_nq * 2]);
  const T G3 = non_temp_load(&G_entity[gid + cube_nq * 3]);
  const T G4 = non_temp_load(&G_entity[gid + cube_nq * 4]);
  const T G5 = non_temp_load(&G_entity[gid + cube_nq * 5]);

  const T coeff = entity_constants[block_id];

  // Store values at quadrature points: scratch2, scratchy, scratchz all
  // have dimensions (nq, nq, nq)
  __syncthreads();
  int idx = ijk<square_nq, nq, 1>(tx, ty, tz);
  scratch1[idx] = coeff * (G0 * val_x + G1 * val_y + G2 * val_z);
  scratch2[idx] = coeff * (G1 * val_x + G3 * val_y + G4 * val_z);
  scratch3[idx] = coeff * (G2 * val_x + G4 * val_y + G5 * val_z);

  // Apply contraction in the x-direction
  // T grad_x = 0;
  // T grad_y = 0;
  // T grad_z = 0;

  // tx is dof index, ty, tz quadrature point indices

  // At this thread's quadrature point, compute r1 = r1_(i0, i1, i2) =
  // \sum_(q0, q2, q3) [\nabla\Phi1_(i0, i1, i2)](q0, q2, q3) \cdot [\nabla
  // u](q0, q2, q3)
  //
  // r1_(i0, i1, i2)
  //    = (d\Phi1_(i0, i1, i2)/dx0)(q0, q1, q2) (du/dx0)(q0, q1, q2)
  //    + (d\Phi1_(i0, i1, i2)/dx1)(q0, q1, q2) (du/dx1)(q0, q1, q2)
  //    + (d\Phi1_(i0, i1, i2)/dx2)(q0, q1, q2) (du/dx2)(q0, q1, q2)
  //
  //    = (d\phi1_(i0)/dx)(q0) \phi1_(i1)(q1) \phi1_(i2)(q2) (du/dx0)(q0, q1,
  //    q2)
  //    + \phi1_(i0)(q0) (d\phi1_(i1)/dx)(q1) \phi1_(i2)(q2) (du/dx1)(q0, q1,
  //    q2)
  //    + \phi1_(i0)(q0) \phi1_(i1)(q1) (d\phi1_(i2)/dx)(q2) (du/dx2)(q0, q1,
  //    q2)
  //
  //    = (d\phi1_(i0)/dx)(q0) \delta_(i1, q1) \delta_(i2, q2) (du/dx0)(q0, q1,
  //    q2)
  //    + \delta_(i0, q0) (d\phi1_(i1)/dx)(q1) \delta_(i2, q2) (du/dx1)(q0, q1,
  //    q2)
  //    + \delta_(i0, q0) \delta_(i1, q1) (d\phi1_(i2)/dx)(q2) (du/dx2)(q0, q1,
  //    q2)
  //
  //    = d\phi1_(i0)/dx(q0) (du/dx0)(q0, i1, i2) + (d\phi1_(i1)/dx)(q1)
  //    (du/dx1)(i0, q1, i2) + (d\phi1_(i2)/dx)(q2) (du/dx2)(i0, i1, q2)
  __syncthreads();
  T yd = 0;
  for (int idx = 0; idx < nq; ++idx)
  {
    yd += dphi1[idx * nq1 + tx] * scratch1[ijk<square_nq, nq, 1>(idx, ty, tz)];
    yd += dphi1[idx * nq1 + ty] * scratch2[ijk<square_nq, nq, 1>(tx, idx, tz)];
    yd += dphi1[idx * nq1 + tz] * scratch3[ijk<square_nq, nq, 1>(tx, ty, idx)];
  }

  // Interpolate quadrature points to dofs
  if (identity != 1)
  {
    // Note that v1 = Pi v0 (v1, v0 are dofs), i.e. v1_(i) = Pi_(i, j) v0_(j),
    // where \Pi_(ij) = \Phi0_(j)(xi) and \Phi0_(i)(xj) is \Phi0_(i)
    // evaluated at node the node of \Phi1_(j). Therefore
    //
    //  vh(x) = \Phi0_(j)(x) v0_j
    //        = \Phi1_(j)(x) \Pi_(ji) v0_i
    //        = \Phi1_(j)(x) \Phi0_(i)(xj) v0_i
    //        = \Phi1_(j)(x) \phi0_(i0)(xj_0) \phi0_(i1)(xj_1j) \phi0_(i2)(xj_2)
    //        v0_(i0, i1, i2)
    //
    // hence
    //
    //  \Phi0(i)(x) = \Phi1_(j)(x) \phi0_(i0)(x_(j0)) \phi0_(i1)(x_(j1))
    //  \phi0_(i2)(x_(j2))
    //
    // and
    //
    //  \Phi0_(i0, i1, i2)(x0, x1, x2) = \phi1_(j0)(x0) \phi1_(j1)(x1)
    //  \phi1_(j2)(x2)
    //         \phi0_(i0)(x_(j0)) \phi0_(i1)(x_(j1)) \phi0_(i2)(x_(j2))
    //
    // Hence:
    //
    //  (d\Phi0_(i0, i1, i2)/dx0)(x0, x1, x2)
    //     = (d\phi1_(j0)/dx)(x0) \phi1_(j1)(x1) \phi1_(j2)(x2)
    //     \phi0_(i0)(x_(j0))
    //          \phi0_(i1)(x_(j1)) \phi0_(i2)(x_(j2))
    //
    // At quadrature points,
    //
    //  (d\Phi0_(i0, i1, i2)/dx0)(q0, q1, q2)
    //     = (d\phi1_(j0)/dx)(q0) \phi1_(j1)(q1) \phi1_(j2)(q2)
    //     \phi0_(i0)(x_(j0))
    //     \phi0_(i1)(x_(j1)) \phi0_(i2)(x_(j2)) = (d\phi1_(j0)/dx)(q0)
    //     \phi0_(i0)(x_(j0))
    //     \phi0_(i1)(x_(j1)) \phi0_(i2)(x_(j2))
    //
    // We want to compute
    //
    // r0_(i0, i1, i2) = (d\Phi0_(i0, i1, i2)/dx0)(q0, q1, q2) (du/dx0)(q0, q1,
    // q2)
    //                 + (d\Phi0_(i0, i1, i2)/dx1)(q0, q1, q2) (du/dx1)(q0, q1,
    //                 q2)
    //                 + (d\Phi0_(i0, i1, i2)/dx2)(q0, q1, q2) (du/dx2)(q0, q1,
    //                 q2)
    //
    //                 = (d\phi1_(q0)/dx)(q0) \phi0_(i0)(q0) \phi0_(i1)(q1)
    //                 \phi0_(i2)(q2) (du/dx0)(q0, q1, q2)
    //                 + (d\phi1_(q1)/dx)(q1) \phi0_(i0)(q0) \phi0_(i1)(q1)
    //                 \phi0_(i2)(q2) (du/dx1)(q0, q1, q2)
    //                 + (d\phi1_(q2)/dx)(q2) \phi0_(i0)(q0) \phi0_(i1)(q1)
    //                 \phi0_(i2)(q2) (du/dx2)(q0, q1, q2)
    //
    //                 = \phi0_(i0)(q0) \phi0_(i1)(q1) \phi0_(i2)(q2)
    //                 [(d\phi1_(q0)/dx)(q0) du/dx0_(q0, q1, q2)
    //                     + (d\phi1_(q1)/dx)(q1) du/dx1_(q0, q1, q2) +
    //                     (d\phi1_(q2)/dx)(q2) du/dx2_(q0, q1, q2)]
    //
    // Have already computed
    //
    // r1_(q0, q1, q2)  = d\phi1_(q0)/dx(q0) du/dx0_(q0, i1, i2) +
    // d\phi1_(q1)/dx(q1) du/dx1_(i0, q1, i2) + d\phi1_(q2)/dx(q2) du/dx2_(i0,
    // i1, q2)
    //
    // So we compute:
    //
    //  r0_(i0, i1, i2) = \phi0_(i0)(q0) \phi0_(i1)(q1) \phi0_(i2)(q2)
    //  [(d\phi1_(q0)/dx)(q0) du/dx0_(q0, q1, q2) r1_(q0, q1, q2)

    __syncthreads();
    scratch1[ijk<square_nq, nq, 1>(tx, ty, tz)] = yd;

    __syncthreads();
    yd = 0;
    if (tx < nd)
    {
      // tmp0(i0, q1, q2) += phi0_(i0)(q0) * r1(q0, q1, q2)
      for (int ix = 0; ix < nq; ++ix)
        yd += phi0[ix * nd + tx] * scratch1[ijk<square_nq, nq, 1>(ix, ty, tz)];
    }

    scratch2[ijk<square_nq, nq, 1>(tx, ty, tz)] = yd;
    __syncthreads();

    yd = 0;
    if (ty < nd)
    {
      // tmp1(i0, i1, q2) += phi0_(i1)(q1) * tmp0(i0, q1, q2)
      for (int iy = 0; iy < nq; ++iy)
        yd += phi0[iy * nd + ty] * scratch2[ijk<square_nq, nq, 1>(tx, iy, tz)];
    }

    scratch3[ijk<square_nq, nq, 1>(tx, ty, tz)] = yd;
    __syncthreads();

    yd = 0;
    if (tz < nd)
    {
      // b(i0, i1, i2) += phi0_(i2)(q2) * tmp1(i0, i1, q2)
      for (int iz = 0; iz < nq; ++iz)
        yd += phi0[iz * nd + tz] * scratch3[ijk<square_nq, nq, 1>(tx, ty, iz)];
    }
  } // end of interpolation

  // Write back to global memory
  if (dof != -1)
    atomicAdd(&b[dof], yd);
}

template <typename T, int P>
__global__ void mat_diagonal(const T* entity_constants, T* b, const T* G_entity,
                             const std::int32_t* entity_dofmap,
                             const int* entities, int n_entities,
                             const std::int8_t* bc_marker)
{
  constexpr int nd = P + 1; // Number of dofs per direction in 1D
  constexpr int nq
      = nd; // Number of quadrature points in 1D (must be the same as nd)

  assert(blockDim.x == nd);
  assert(blockDim.y == nd);
  assert(blockDim.z == nd);

  constexpr int cube_nd = nd * nd * nd;
  constexpr int square_nd = nd * nd;

  int tx = threadIdx.x; // 1d dofs x direction
  int ty = threadIdx.y; // 1d dofs y direction
  int tz = threadIdx.z; // 1d dofs z direction

  // thread_id represents the dof index in 3D
  int thread_id = tx * square_nd + ty * nd + tz;
  // block_id is the cell (or facet) index
  int block_id = blockIdx.x;

  // Check if the block_id is valid (i.e. within the number of entities)
  if (block_id >= n_entities)
    return;

  // Get transform at quadrature point (thread)
  constexpr int nq3 = nq * nq * nq;

  int offset = (block_id * nq3 * 6) + thread_id;
  T G1 = G_entity[offset + 1 * nq3];
  T G2 = G_entity[offset + 2 * nq3];
  T G4 = G_entity[offset + 4 * nq3];

  // DG-0 Coefficient
  T coeff = entity_constants[block_id];

  T val = 0.0;

  auto dphi_
      = [&](int ix, int iy) { return dphi1_const<T, P + 1>[ix * nq + iy]; };
  auto G_ = [&](int r, int ix, int iy, int iz)
  {
    int qid = ix * square_nd + iy * nd + iz;
    const int offset = block_id * cube_nd * 6 + qid;
    return G_entity[offset + r * cube_nd];
  };

  for (int iq0 = 0; iq0 < nq; ++iq0)
    val += G_(0, iq0, ty, tz) * dphi_(iq0, tx) * dphi_(iq0, tx);
  for (int iq1 = 0; iq1 < nq; ++iq1)
    val += G_(3, tx, iq1, tz) * dphi_(iq1, ty) * dphi_(iq1, ty);
  for (int iq2 = 0; iq2 < nq; ++iq2)
    val += G_(5, tx, ty, iq2) * dphi_(iq2, tz) * dphi_(iq2, tz);

  val += G1 * dphi_(tx, tx) * dphi_(ty, ty);
  val += G2 * dphi_(tx, tx) * dphi_(tz, tz);
  val += G1 * dphi_(ty, ty) * dphi_(tx, tx);
  val += G4 * dphi_(ty, ty) * dphi_(tz, tz);
  val += G2 * dphi_(tz, tz) * dphi_(tx, tx);
  val += G4 * dphi_(tz, tz) * dphi_(ty, ty);

  int dof = entity_dofmap[entities[block_id] * cube_nd + thread_id];
  if (bc_marker[dof])
    b[dof] = T(1.0);
  else
    atomicAdd(&b[dof], coeff * val);
}

namespace dolfinx::acc
{

// FIXME Could just replace these maps with expression
const std::map<int, int> q_map_gll
    = {{1, 1}, {2, 3}, {3, 4}, {4, 6}, {5, 8}, {6, 10}, {7, 12}, {8, 14}};

const std::map<int, int> q_map_gq
    = {{1, 2}, {2, 4}, {3, 6}, {4, 8}, {5, 10}, {6, 12}, {7, 14}, {8, 16}};

template <typename T>
class MatFreeLaplacian
{
public:
  using value_type = T;

  MatFreeLaplacian(int degree, int qmode, std::span<const T> coefficients,
                   std::span<const std::int32_t> dofmap,
                   std::span<const T> xgeom,
                   std::span<const std::int32_t> geometry_dofmap,
                   const dolfinx::fem::CoordinateElement<T>& cmap,
                   const std::vector<int>& lcells,
                   const std::vector<int>& bcells,
                   std::span<const std::int8_t> bc_marker,
                   basix::quadrature::type quad_type,
                   std::size_t batch_size = 0)
      : degree(degree), cell_constants(coefficients), cell_dofmap(dofmap),
        xgeom(xgeom), geometry_dofmap(geometry_dofmap), bc_marker(bc_marker),
        batch_size(batch_size)
  {
    basix::element::lagrange_variant variant;
    std::map<int, int> q_map;
    if (quad_type == basix::quadrature::type::gauss_jacobi)
    {
      variant = basix::element::lagrange_variant::gl_warped;
      q_map = q_map_gq;
    }
    else if (quad_type == basix::quadrature::type::gll)
    {
      variant = basix::element::lagrange_variant::gll_warped;
      q_map = q_map_gll;
    }
    else
      throw std::runtime_error(
          "Unsupported quadrature type for mat-free operator");

    // NOTE: Basix generates quadrature points in tensor-product ordering, so
    // this is OK
    auto [Gpoints, Gweights] = basix::quadrature::make_quadrature<T>(
        quad_type, basix::cell::type::hexahedron,
        basix::polyset::type::standard, q_map.at(degree + qmode));

    std::array<std::size_t, 4> phi_shape
        = cmap.tabulate_shape(1, Gweights.size());
    std::vector<T> phi_b(
        std::reduce(phi_shape.begin(), phi_shape.end(), 1, std::multiplies{}));
    cmap.tabulate(1, Gpoints, {Gweights.size(), 3}, phi_b);

    // Copy dphi to device (skipping phi in table)
    dphi_geometry.resize(phi_b.size() * 3 / 4);
    thrust::copy(phi_b.begin() + phi_b.size() / 4, phi_b.end(),
                 dphi_geometry.begin());

    Gweights_d.resize(Gweights.size());
    thrust::copy(Gweights.begin(), Gweights.end(), Gweights_d.begin());

    // Create 1D element
    basix::FiniteElement<T> element0 = basix::create_element<T>(
        basix::element::family::P, basix::cell::type::interval, degree,
        basix::element::lagrange_variant::gll_warped,
        basix::element::dpc_variant::unset, false);

    // Create quadrature
    auto [points, weights] = basix::quadrature::make_quadrature<T>(
        quad_type, basix::cell::type::interval, basix::polyset::type::standard,
        q_map.at(degree + qmode));

    // Make sure geometry weights for 3D cell match size of 1D
    // quadrature weights
    op_nq = weights.size();
    if (Gweights.size() != op_nq * op_nq * op_nq)
      throw std::runtime_error("3D and 1D weight mismatch");

    // Create higher-order 1D element for which the dofs coincide with
    // the quadrature points
    basix::FiniteElement<T> element1 = basix::create_element<T>(
        basix::element::family::P, basix::cell::type::interval, op_nq - 1,
        variant, basix::element::dpc_variant::unset, true);

    // Compute interpolation matrix from element0 to element1
    auto [mat, shape_I]
        = basix::compute_interpolation_operator(element0, element1);
    for (int i = 0; i < shape_I[0]; ++i)
    {
      std::cout << "[ ";
      for (int j = 0; j < shape_I[1]; ++j)
        std::cout << std::fixed << std::setw(8) << mat[i * shape_I[1] + j]
                  << " ";
      std::cout << "]\n";
      std::cout << std::defaultfloat;
    }

    T precision = std::numeric_limits<T>::epsilon();
    for (auto& v : mat)
    {
      if (std::abs(v) < 5 * precision)
        v = 0;
    }

    // Check whether the interpolation matrix is the identity
    is_identity = matrix_is_identity(mat, shape_I);

    spdlog::info("Identity: {}", is_identity);

    // Tabulate 1D
    auto [table, shape] = element1.tabulate(1, points, {weights.size(), 1});

    // Basis value gradient evualation table
    if (op_nq == degree + 1)
    {
      switch (degree)
      {
      case 1:
        copy_phi_tables<1, 2>(mat, table);
        break;
      case 2:
        copy_phi_tables<2, 3>(mat, table);
        break;
      case 3:
        copy_phi_tables<3, 4>(mat, table);
        break;
      case 4:
        copy_phi_tables<4, 5>(mat, table);
        break;
      case 5:
        copy_phi_tables<5, 6>(mat, table);
        break;
      case 6:
        copy_phi_tables<6, 7>(mat, table);
        break;
      case 7:
        copy_phi_tables<7, 8>(mat, table);
        break;
      default:
        throw std::runtime_error("Unsupported degree");
      }
    }
    else if (op_nq == degree + 2)
    {
      switch (degree)
      {
      case 1:
        copy_phi_tables<1, 3>(mat, table);
        break;
      case 2:
        copy_phi_tables<2, 4>(mat, table);
        break;
      case 3:
        copy_phi_tables<3, 5>(mat, table);
        break;
      case 4:
        copy_phi_tables<4, 6>(mat, table);
        break;
      case 5:
        copy_phi_tables<5, 7>(mat, table);
        break;
      case 6:
        copy_phi_tables<6, 8>(mat, table);
        break;
      case 7:
        copy_phi_tables<7, 9>(mat, table);
        break;
      default:
        throw std::runtime_error("Unsupported degree");
      }
    }
    else
      throw std::runtime_error("Unsupported nq");

    // Copy interpolation matrix to device
    spdlog::debug("Copy interpolation matrix to device ({} bytes)",
                  mat.size() * sizeof(T));

    // Copy lists of local and boundary cells to device
    lcells_device.resize(lcells.size());
    thrust::copy(lcells.begin(), lcells.end(), lcells_device.begin());
    bcells_device.resize(bcells.size());
    thrust::copy(bcells.begin(), bcells.end(), bcells_device.begin());

    // If we're not batching the geometry, precompute it
    if (batch_size == 0)
    {
      // FIXME Store cells and local/ghost offsets instead to avoid this?
      spdlog::info("Precomputing geometry");
      thrust::device_vector<std::int32_t> cells_d(lcells_device.size()
                                                  + bcells_device.size());
      thrust::copy(lcells_device.begin(), lcells_device.end(), cells_d.begin());
      thrust::copy(bcells_device.begin(), bcells_device.end(),
                   cells_d.begin() + lcells_device.size());
      std::span<std::int32_t> cell_list_d(
          thrust::raw_pointer_cast(cells_d.data()), cells_d.size());

      compute_geometry(op_nq, cell_list_d);
      device_synchronize();
    }

    spdlog::debug("Done MatFreeLaplacian constructor");
  }

  /// @brief Compute weighted geometry data on GPU
  /// @param nq Number of quadrature points in 1D
  /// @param cell_list_d List of cell indices to compute for
  template <int Q = 2>
  void compute_geometry(int nq, std::span<int> cell_list_d)
  {
    if constexpr (Q < 10)
    {
      if (nq > Q)
        compute_geometry<Q + 1>(nq, cell_list_d);
      else
      {
        assert(nq == Q);
        G_entity.resize(Gweights_d.size() * cell_list_d.size() * 6);
        dim3 block_size(Gweights_d.size());
        dim3 grid_size(cell_list_d.size());

        spdlog::info("xgeom size {}", xgeom.size());
        spdlog::info("G_entity size {}", G_entity.size());
        spdlog::info("geometry_dofmap size {}", geometry_dofmap.size());
        spdlog::info("dphi_geometry size {}", dphi_geometry.size());
        spdlog::info("Gweights size {}", Gweights_d.size());
        spdlog::info("cell_list_d size {}", cell_list_d.size());
        spdlog::info("Calling geometry_computation [{} {}]", Q, nq);

        std::size_t shm_size = 24 * sizeof(T); // coordinate size (8x3)
        geometry_computation<T, Q><<<grid_size, block_size, shm_size, 0>>>(
            xgeom.data(), thrust::raw_pointer_cast(G_entity.data()),
            geometry_dofmap.data(),
            thrust::raw_pointer_cast(dphi_geometry.data()),
            thrust::raw_pointer_cast(Gweights_d.data()), cell_list_d.data(),
            cell_list_d.size());
        spdlog::debug("Done geometry_computation");
      }
    }
    else
      throw std::runtime_error("Unsupported degree [geometry]");
  }

  /// @brief Compute matrix diagonal entries
  /// @param out Vector containing diagonal entries of the matrix
  /// @note Only works for qmode=0, i.e. when Q=P+1
  template <int P, typename Vector>
  void compute_mat_diag_inv(Vector& out)
  {
    if (!lcells_device.empty())
    {
      spdlog::debug("mat_diagonal doing lcells. lcells size = {}",
                    lcells_device.size());
      std::span<int> cell_list_d(thrust::raw_pointer_cast(lcells_device.data()),
                                 lcells_device.size());
      compute_geometry(P + 1, cell_list_d);
      device_synchronize();

      out.set(T{0.0});
      T* y = out.mutable_array().data();

      dim3 block_size(P + 1, P + 1, P + 1);
      dim3 grid_size(cell_list_d.size());
      spdlog::debug("Calling mat_diagonal");
      mat_diagonal<T, P><<<grid_size, block_size, 0>>>(
          cell_constants.data(), y, thrust::raw_pointer_cast(G_entity.data()),
          cell_dofmap.data(), cell_list_d.data(), cell_list_d.size(),
          bc_marker.data());
      check_device_last_error();
    }

    if (!bcells_device.empty())
    {
      spdlog::debug("mat_diagonal doing bcells. bcells size = {}",
                    bcells_device.size());
      std::span<int> cell_list_d(thrust::raw_pointer_cast(bcells_device.data()),
                                 bcells_device.size());
      compute_geometry(P + 1, cell_list_d);
      device_synchronize();

      T* y = out.mutable_array().data();

      dim3 block_size(P + 1, P + 1, P + 1);
      dim3 grid_size(cell_list_d.size());
      mat_diagonal<T, P><<<grid_size, block_size, 0>>>(
          cell_constants.data(), y, thrust::raw_pointer_cast(G_entity.data()),
          cell_dofmap.data(), cell_list_d.data(), cell_list_d.size(),
          bc_marker.data());
      check_device_last_error();
    }

    // Invert
    thrust::transform(thrust::device, out.array().begin(),
                      out.array().begin() + out.map()->size_local(),
                      out.mutable_array().begin(),
                      [] __host__ __device__(T yi) { return 1.0 / yi; });
  }

  template <int P, int Q, typename Vector>
  void impl_operator(Vector& in, Vector& out)
  {
    spdlog::debug("impl_operator operator start");

    in.scatter_fwd_begin();

    T* geometry_ptr = thrust::raw_pointer_cast(G_entity.data());

    if (!lcells_device.empty())
    {
      std::size_t i = 0;
      std::size_t i_batch_size
          = (batch_size == 0) ? lcells_device.size() : batch_size;
      while (i < lcells_device.size())
      {
        std::size_t i_next = std::min(lcells_device.size(), i + i_batch_size);
        std::span<int> cell_list_d(
            thrust::raw_pointer_cast(lcells_device.data()) + i, (i_next - i));
        i = i_next;

        if (batch_size > 0)
        {
          spdlog::debug("Calling compute_geometry on local cells [{}]",
                        cell_list_d.size());
          compute_geometry(Q, cell_list_d);
          device_synchronize();
        }

        spdlog::debug("Calling stiffness_operator on local cells [{}]",
                      cell_list_d.size());
        T* x = in.mutable_array().data();
        T* y = out.mutable_array().data();

#ifdef USE_SLICED
        constexpr int quad_per_thread = SLICE_SIZE;
        dim3 block_size(Q, Q, (Q + quad_per_thread - 1) / quad_per_thread);
        dim3 grid_size(cell_list_d.size());
        sliced::stiffness_operator<T, P, Q, quad_per_thread>
            <<<grid_size, block_size>>>(x, cell_constants.data(), y,
                                        geometry_ptr, cell_dofmap.data(),
                                        cell_list_d.data(), cell_list_d.size(),
                                        bc_marker.data(), is_identity);
#else
        dim3 block_size(Q, Q, Q);
        dim3 grid_size(cell_list_d.size());
        stiffness_operator<T, P, Q><<<grid_size, block_size>>>(
            x, cell_constants.data(), y, geometry_ptr, cell_dofmap.data(),
            cell_list_d.data(), cell_list_d.size(), bc_marker.data(),
            is_identity);
#endif

        check_device_last_error();
      }
    }

    spdlog::debug("impl_operator done lcells");

    spdlog::debug("cell_constants size {}", cell_constants.size());
    spdlog::debug("in size {}", in.array().size());
    spdlog::debug("out size {}", out.array().size());
    spdlog::debug("G_entity size {}", G_entity.size());
    spdlog::debug("cell_dofmap size {}", cell_dofmap.size());
    spdlog::debug("bc_marker size {}", bc_marker.size());

    in.scatter_fwd_end();

    spdlog::debug("impl_operator after scatter");

    if (!bcells_device.empty())
    {
      spdlog::debug("impl_operator doing bcells. bcells size = {}",
                    bcells_device.size());
      std::span<int> cell_list_d(thrust::raw_pointer_cast(bcells_device.data()),
                                 bcells_device.size());

      if (batch_size > 0)
      {
        compute_geometry(Q, cell_list_d);
        device_synchronize();
      }
      else
        geometry_ptr += 6 * Q * Q * Q * lcells_device.size();

      T* x = in.mutable_array().data();
      T* y = out.mutable_array().data();

      // FIXME: Should have NQ as a template parameter
#ifdef USE_SLICED
      constexpr int quad_per_thread = SLICE_SIZE;
      dim3 block_size(Q, Q, (Q + quad_per_thread - 1) / quad_per_thread);
      dim3 grid_size(cell_list_d.size());
      sliced::stiffness_operator<T, P, Q, quad_per_thread>
          <<<grid_size, block_size>>>(x, cell_constants.data(), y, geometry_ptr,
                                      cell_dofmap.data(), cell_list_d.data(),
                                      cell_list_d.size(), bc_marker.data(),
                                      is_identity);
#else
      dim3 block_size(Q, Q, Q);
      dim3 grid_size(cell_list_d.size());
      stiffness_operator<T, P, Q><<<grid_size, block_size>>>(
          x, cell_constants.data(), y, geometry_ptr, cell_dofmap.data(),
          cell_list_d.data(), cell_list_d.size(), bc_marker.data(),
          is_identity);
#endif
      check_device_last_error();
    }

    device_synchronize();

    spdlog::debug("impl_operator done bcells");
  }

  /// @brief Apply Laplacian operator
  /// @param in Input vector
  /// @param out Output vector
  template <typename Vector>
  void operator()(Vector& in, Vector& out)
  {
    spdlog::debug("Mat free operator start");
    out.set(T{0.0});

    if (op_nq == degree + 1)
    {
      if (degree == 1)
        impl_operator<1, 2>(in, out);
      else if (degree == 2)
        impl_operator<2, 3>(in, out);
      else if (degree == 3)
        impl_operator<3, 4>(in, out);
      else if (degree == 4)
        impl_operator<4, 5>(in, out);
      else if (degree == 5)
        impl_operator<5, 6>(in, out);
      else if (degree == 6)
        impl_operator<6, 7>(in, out);
      else if (degree == 7)
        impl_operator<7, 8>(in, out);
      else
        throw std::runtime_error("Unsupported degree [operator]");
    }
    else if (op_nq == degree + 2)
    {
      if (degree == 1)
        impl_operator<1, 3>(in, out);
      else if (degree == 2)
        impl_operator<2, 4>(in, out);
      else if (degree == 3)
        impl_operator<3, 5>(in, out);
      else if (degree == 4)
        impl_operator<4, 6>(in, out);
      else if (degree == 5)
        impl_operator<5, 7>(in, out);
      else if (degree == 6)
        impl_operator<6, 8>(in, out);
      else if (degree == 7)
        impl_operator<7, 9>(in, out);
      else
        throw std::runtime_error("Unsupported degree [operator]");
    }
    else
    {
      throw std::runtime_error("Unsupported nq");
    }

    spdlog::debug("Mat free operator end");
  }

  template <typename Vector>
  void get_diag_inverse(Vector& diag_inv)
  {
    spdlog::debug("Mat diagonal operator start");

    if (degree == 1)
      compute_mat_diag_inv<1>(diag_inv);
    else if (degree == 2)
      compute_mat_diag_inv<2>(diag_inv);
    else if (degree == 3)
      compute_mat_diag_inv<3>(diag_inv);
    else if (degree == 4)
      compute_mat_diag_inv<4>(diag_inv);
    else if (degree == 5)
      compute_mat_diag_inv<5>(diag_inv);
    else if (degree == 6)
      compute_mat_diag_inv<6>(diag_inv);
    else if (degree == 7)
      compute_mat_diag_inv<7>(diag_inv);
    else
      throw std::runtime_error("Unsupported degree [mat diag]");

    spdlog::debug("Mat diagonal operator end");
  }

private:
  int degree;

  // Number of quadrature points in 1D
  int op_nq;

  // Reference to on-device storage for constants, dofmap etc.
  std::span<const T> cell_constants;
  std::span<const std::int32_t> cell_dofmap;

  // Reference to on-device storage of geometry data
  std::span<const T> xgeom;
  std::span<const std::int32_t> geometry_dofmap;

  // geometry tables dphi on device
  thrust::device_vector<T> dphi_geometry;

  std::span<const std::int8_t> bc_marker;

  // On device storage for geometry quadrature weights
  thrust::device_vector<T> Gweights_d;

  // On device storage for geometry data (computed for each batch of cells)
  thrust::device_vector<T> G_entity;

  // Interpolation is the identity
  bool is_identity;

  // Lists of cells which are local (lcells) and boundary (bcells)
  thrust::device_vector<int> lcells_device, bcells_device;

  // On device storage for the inverse diagonal, needed for Jacobi
  // preconditioner (to remove in future)
  thrust::device_vector<T> _diag_inv;

  // Batch size for geometry computation (set to 0 for no batching)
  std::size_t batch_size;

  template <int P, int Q>
  void copy_phi_tables(std::span<const T> phi0, std::span<const T> dphi1)
  {
    err_check(deviceMemcpyToSymbol((phi0_const<T, P, Q>), phi0.data(),
                                   phi0.size() * sizeof(T)));
    err_check(deviceMemcpyToSymbol((dphi1_const<T, Q>),
                                   dphi1.data() + dphi1.size() / 2,
                                   (dphi1.size() / 2) * sizeof(T)));
  }
};

} // namespace dolfinx::acc

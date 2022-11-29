// Copyright (C) 2021 Igor A. Baratta, Chris Richardson
// SPDX-License-Identifier:    MIT

#include "cublas_v2.h"
#include <algorithm>
#include <cuda_runtime.h>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/la/Vector.h>
#include <type_traits>

using namespace dolfinx;

namespace
{
template <class T>
struct dependent_false : std::false_type
{
};

template <typename C>
void assert_cuda(C e)
{
  if (e != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error("CUDA ERROR: " + std::to_string(e));
}
} // namespace

namespace linalg
{
/// Compute vector y = alpha*x + y
/// @param[in, out] y Result
/// @param[in] alpha
/// @param[in] x
template <typename Scalar, typename Vector>
void axpy(cublasHandle_t handle, Scalar alpha, const Vector& x, Vector& y)
{
  using T = typename Vector::value_type;

  assert(x.array().size() == y.array().size());
  const T* _x = x.array().data();
  T* _y = y.mutable_array().data();
  std::size_t n = x.map()->size_local();

  cublasStatus_t status;
  T _alpha = static_cast<T>(alpha);

  if constexpr (std::is_same<T, double>())
    status = cublasDaxpy(handle, n, &_alpha, _x, 1, _y, 1);
  else if constexpr (std::is_same<T, float>())
    status = cublasSaxpy(handle, n, &alpha, _x, 1, _y, 1);
  else
    static_assert(dependent_false<T>::value);

  assert_cuda(status);
  cudaDeviceSynchronize();
}

template <typename Vector>
Vector::value_type inner_product(cublasHandle_t handle, Vector& x, Vector& y)
{
  using T = typename Vector::value_type;
  assert(x.array().size() == y.array().size());

  const T* _x = x.array().data();
  const T* _y = y.array().data();
  std::size_t n = x.map()->size_local();

  T result = 0;

  cublasStatus_t status;

  if constexpr (std::is_same<T, double>())
    status = cublasDdot(handle, n, _x, 1, _y, 1, &result);
  else if constexpr (std::is_same<T, float>())
    status = cublasSdot(handle, n, _x, 1, _y, 1, &result);
  else
    static_assert(dependent_false<T>::value);
  assert_cuda(status);
  cudaDeviceSynchronize();
  return result;
}

template <typename Scalar, typename Vector>
void scale(cublasHandle_t handle, Scalar alpha, Vector& x)
{
  using T = typename Vector::value_type;
  T* _x = x.mutable_array().data();
  std::size_t n = x.map()->size_local();

  T _alpha = static_cast<T>(alpha);
  cublasStatus_t status;

  if constexpr (std::is_same<T, double>())
    status = cublasDscal(handle, n, &_alpha, _x, 1);
  else if constexpr (std::is_same<T, float>())
    status = cublasSscal(handle, n, &_alpha, _x, 1);
  else
    static_assert(dependent_false<T>::value);
  assert_cuda(status);
  cudaDeviceSynchronize();
}

template <typename Vector1, typename Vector2>
void copy(cublasHandle_t handle, const Vector1& x, Vector2& y)
{

  using T = typename Vector1::value_type;
  assert(x.array().size() == y.array().size());

  const T* _x = x.array().data();
  T* _y = y.mutable_array().data();
  std::size_t n = x.map()->size_local();

  cublasStatus_t status;

  if constexpr (std::is_same<T, double>())
    status = cublasDcopy(handle, n, _x, 1, _y, 1);
  else if constexpr (std::is_same<T, float>())
    status = cublasScopy(handle, n, _x, 1, _y, 1);
  else
    static_assert(dependent_false<T>::value);

  assert_cuda(status);
  cudaDeviceSynchronize();
}

template <typename Vector>
void print(const Vector& x, int n = 0)
{
  auto array = x.array();
  if (n == 0)
    n = array.size();

  std::for_each_n(array.begin(), n, [](auto e) { std::cout << e << " "; });
  std::cout << std::endl;
}

/// Solve problem A.x = b using the Conjugate Gradient method
/// @tparam U The scalar type
/// @tparam ApplyFunction Type of the function object "action"
/// @param[in, out] x Solution vector, may be set to an initial guess
/// @param[in] b RHS Vector
/// @param[in] action Function that provides the action of the linear operator
/// @param[in] kmax Maximum number of iterations
/// @param[in] rtol Relative tolerances for convergence
/// @return The number if iterations
/// @pre It is required that the ghost values of `x` and `b` have been
/// updated before this function is called
template <typename Vector, typename ApplyFunction>
int cg(cublasHandle_t handle, Vector& x, const Vector& b,
       ApplyFunction&& action, int kmax = 50, double rtol = 1e-8)
{

  using T = Vector::value_type;

  // Create working vectors
  Vector r(b), y(b);

  // Compute initial residual r0 = b - Ax0
  action(x, y);              // y = Ax0
  axpy(handle, T(-1), r, y); // r = (-1)*y + b

  // Create p work vector
  Vector p(r);

  // Iterations of CG
  auto rnorm0 = inner_product(handle, r, r);
  const auto rtol2 = rtol * rtol;
  auto rnorm = rnorm0;
  int k = 0;
  while (k < kmax)
  {
    ++k;

    // Compute y = A p
    action(p, y);

    // Compute alpha = r.r/p.y
    const T alpha = rnorm / inner_product(handle, p, y);

    // Update x (x <- x + alpha*p)
    axpy(handle, alpha, x, p);

    // Update r (r <- r - alpha*y)
    axpy(handle, -alpha, r, y);

    // Update residual norm
    const auto rnorm_new = inner_product(handle, r, r);
    const T beta = rnorm_new / rnorm;
    rnorm = rnorm_new;

    std::cout << "alpha :" << alpha << std::endl;
    std::cout << "beta :" << beta << std::endl;
    std::cout << "rnorm :" << rnorm << std::endl;


    if (rnorm / rnorm0 < rtol2)
      break;

    // Update p (p <- beta*p + r)
    // p = beta*p
    scale(handle, beta, p);
    // p = p + r
    axpy(handle, T(1), p, r);
  }

  return k;
}
} // namespace linalg

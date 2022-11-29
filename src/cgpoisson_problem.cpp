// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include "cgpoisson_problem.h"
#include "Poisson.h"
#include "cuda_allocator.h"
#include "solvers/cg.h"
#include "solvers/poisson.hpp"
#include "solvers/scatter.hpp"
#include <cfloat>
#include <cmath>
#include <dolfinx/common/Scatterer.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <memory>
#include <petscsys.h>
#include <utility>

using namespace dolfinx;
using T = PetscScalar;

using cuda_vector = la::Vector<PetscScalar, CUDA::allocator<PetscScalar>>;

std::tuple<std::shared_ptr<cuda_vector>, std::shared_ptr<cuda_vector>,
           std::function<int(cuda_vector&, const cuda_vector&)>>
cgpoisson::problem(std::shared_ptr<mesh::Mesh> mesh, int order,
                   std::string scatterer)
{
  MPI_Comm comm = mesh->comm();
  int rank = dolfinx::MPI::rank(comm);

  int num_gpus = 0;
  cudaGetDeviceCount(&num_gpus);

  cublasHandle_t handle;
  cublasCreate(&handle);
  cudaSetDevice(rank);

  common::Timer t0("ZZZ FunctionSpace");
  std::vector fs_poisson_a
      = {functionspace_form_Poisson_a1, functionspace_form_Poisson_a2,
         functionspace_form_Poisson_a3};

  auto V = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(*fs_poisson_a.at(order - 1), "v_0", mesh));

  t0.stop();

  common::Timer t1("ZZZ Assemble");

  common::Timer t2("ZZZ Create boundary conditions");
  // Define boundary condition
  auto u0 = std::make_shared<fem::Function<T>>(V);
  u0->x()->set(0);

  // Find facets with bc applied
  const int tdim = mesh->topology().dim();
  const std::vector<std::int32_t> bc_facets = mesh::locate_entities(
      *mesh, tdim - 1,
      [](auto x)
      {
        constexpr double eps = 1.0e-8;
        std::vector<std::int8_t> marker(x.extent(1), false);
        for (std::size_t p = 0; p < x.extent(1); ++p)
        {
          double x0 = x(0, p);
          if (std::abs(x0) < eps or std::abs(x0 - 1) < eps)
            marker[p] = true;
        }
        return marker;
      });

  // Find constrained dofs
  const std::vector<std::int32_t> bdofs
      = fem::locate_dofs_topological(*V, tdim - 1, bc_facets);

  auto bc = std::make_shared<fem::DirichletBC<T>>(u0, bdofs);
  t2.stop();

  // Define coefficients
  common::Timer t3("ZZZ Create RHS function");
  auto f = std::make_shared<fem::Function<T>>(V);
  auto g = std::make_shared<fem::Function<T>>(V);
  f->interpolate(
      [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
      {
        std::vector<T> v(x.extent(1));
        for (std::size_t p = 0; p < x.extent(1); ++p)
        {
          double dx = x(0, p) - 0.5;
          double dy = x(1, p) - 0.5;
          double dr = dx * dx + dy * dy;
          v[p] = 10 * std::exp(-dr / 0.02);
        }

        return {std::move(v), {v.size()}};
      });
  g->interpolate(
      [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
      {
        std::vector<T> f(x.extent(1));
        for (std::size_t p = 0; p < x.extent(1); ++p)
          f[p] = std::sin(5 * x(0, p));
        return {f, {f.size()}};
      });
  t3.stop();

  std::vector form_poisson_L
      = {form_Poisson_L1, form_Poisson_L2, form_Poisson_L3};
  std::vector form_poisson_a
      = {form_Poisson_a1, form_Poisson_a2, form_Poisson_a3};
  std::vector form_poisson_M
      = {form_Poisson_M1, form_Poisson_M2, form_Poisson_M3};

  // Define variational forms
  auto L = std::make_shared<fem::Form<T>>(fem::create_form<T>(
      *form_poisson_L.at(order - 1), {V}, {{"w0", f}, {"w1", g}}, {}, {}));

  auto un = std::make_shared<fem::Function<T>>(V);
  auto M = std::make_shared<fem::Form<T>>(fem::create_form<T>(
      *form_poisson_M.at(order - 1), {V}, {{"w0", un}}, {{}}, {}));

  // Create la::Vector
  CUDA::allocator<PetscScalar> data_allocator;

  auto map = L->function_spaces()[0]->dofmap()->index_map;
  auto bs = L->function_spaces()[0]->dofmap()->index_map_bs();
  cuda_vector b(map, bs, data_allocator);
  cuda_vector b_(map, bs, data_allocator);

  common::Timer t5("ZZZ Assemble vector");
  const std::vector constants_L = fem::pack_constants(*L);
  auto coeffs_L = fem::allocate_coefficient_storage(*L);
  fem::pack_coefficients(*L, coeffs_L);
  fem::assemble_vector<T>(b.mutable_array(), *L, constants_L,
                          fem::make_coefficients_span(coeffs_L));

  // Apply lifting to account for Dirichlet boundary condition
  // b <- b - A * x_bc
  fem::set_bc(un->x()->mutable_array(), {bc}, -1.0);
  fem::assemble_vector(b.mutable_array(), *M);

  // Communicate ghost values
  b.scatter_rev(std::plus<T>());

  // Set BC dofs to zero (effectively zeroes columns of A)
  fem::set_bc(b.mutable_array(), {bc}, 0.0);
  b.scatter_fwd();

  // Pack coefficients and constants

  if (un->x()->array().size() != b.array().size())
    throw std::runtime_error("error");
  // Create Function to hold solution
  auto u = std::make_shared<fem::Function<T>>(V);

  std::function<int(cuda_vector&, const cuda_vector&)> solver_function
      = [=](cuda_vector& u, const cuda_vector& b)
  {
    using ind_allocator_t = CUDA::allocator<std::int32_t>;
    ind_allocator_t ind_alloc;

    using data_allocator_t = CUDA::allocator<T>;
    data_allocator_t data_alloc;

    auto idx_map = V->dofmap()->index_map;
    int bs = V->dofmap()->bs();
    common::Scatterer<ind_allocator_t> sct(*idx_map, bs, ind_alloc);
    std::vector<T, data_allocator_t> local_buffer(sct.local_buffer_size(), 0,
                                                  data_alloc);
    std::vector<T, data_allocator_t> remote_buffer(sct.remote_buffer_size(), 0,
                                                   data_alloc);

    auto pack_fn = [](const auto& in, const auto& idx, auto& out)
    {
      //   out[i] = in[idx[i]];
      gather<T>(idx.size(), idx.data(), in.data(), out.data(), 512);
    };
    auto unpack_fn = [](const auto& in, const auto& idx, auto& out, auto op)
    {
      //   out[idx[i]] = op(out[idx[i]], in[i]);
      scatter<T>(idx.size(), idx.data(), in.data(), out.data(), 512);
    };

    common::Scatterer<ind_allocator_t>::type type;
    if (scatterer == "neighbor")
      type = common::Scatterer<ind_allocator_t>::type::neighbor;
    if (scatterer == "p2p")
      type = common::Scatterer<ind_allocator_t>::type::p2p;

    std::vector<MPI_Request> request = sct.create_request_vector(type);

    auto mesh = V->mesh();
    std::int32_t ncells = mesh->topology().index_map(3)->size_local();

    auto dofmap = V->dofmap()->list().array();
    std::vector<std::int32_t, ind_allocator_t> dofmap_dev(dofmap.size(),
                                                          ind_alloc);
    cudaMemcpy(dofmap_dev.data(), dofmap.data(),
               dofmap.size() * sizeof(std::int32_t), cudaMemcpyHostToDevice);

    std::vector<double, data_allocator_t> w(dofmap.size(), data_alloc);
    std::vector<double, data_allocator_t> A(dofmap.size(), data_alloc);

    auto gdmap = mesh->geometry().dofmap().array();
    std::vector<std::int32_t, ind_allocator_t> gdmap_dev(gdmap.size() * 3,
                                                         ind_alloc);

    for (std::size_t i = 0; i < gdmap.size(); i++)
      for (int j = 0; j < 3; j++)
        gdmap_dev[i * 3 + j] = gdmap[i] * 3 + j;

    auto _geom = mesh->geometry().x();
    std::vector<double, data_allocator_t> geometry_data(_geom.size(),
                                                        data_alloc);
    cudaMemcpy(geometry_data.data(), _geom.data(),
               geometry_data.size() * sizeof(double), cudaMemcpyHostToDevice);

    std::vector<double, data_allocator_t> coordinate_dofs(gdmap_dev.size(),
                                                          data_alloc);
    gather(gdmap_dev.size(), gdmap_dev.data(), geometry_data.data(),
           coordinate_dofs.data(), 512);

    int ndofs_cell = V->element()->space_dimension();

    // Create function for computing the action of A on x (y = Ax)
    auto action = [&](cuda_vector& x, cuda_vector& y)
    {
      cudaMemset(A.data(), T(0), A.size() * sizeof(T));
      auto _y = y.mutable_array();
      cudaMemset(_y.data(), T(0), _y.size() * sizeof(T));

      cudaDeviceSynchronize();
      gather(dofmap_dev.size(), dofmap_dev.data(), x.array().data(), w.data(),
             512);
      cudaDeviceSynchronize();
      poisson(ncells, A.data(), w.data(), coordinate_dofs.data(), ndofs_cell,
              512);
      cudaDeviceSynchronize();
      scatter(dofmap_dev.size(), dofmap_dev.data(), A.data(),
              y.mutable_array().data(), 512);
      cudaDeviceSynchronize();

      const std::int32_t local_size = bs * idx_map->size_local();
      const std::int32_t num_ghosts = bs * idx_map->num_ghosts();
      T* y_data = y.mutable_array().data();
      std::span<T> remote_data(y_data + local_size, num_ghosts);
      std::span<T> local_data(y_data, local_size);
      sct.scatter_rev_begin<T>(remote_data, remote_buffer, local_buffer,
                               pack_fn, request, type);
      sct.scatter_rev_end<T>(local_buffer, local_data, unpack_fn,
                             std::plus<T>(), request);

      // Update ghost values
      sct.scatter_fwd_begin<T>(local_data, local_buffer, remote_buffer, pack_fn,
                               request, type);
      sct.scatter_fwd_end<T>(remote_buffer, remote_data, unpack_fn, request);
    };

    int num_it = linalg::cg(handle, u, b, action, 5, 1e-6);
    return num_it;
  };

  return {std::make_shared<cuda_vector>(std::move(b)),
          std::make_shared<cuda_vector>(std::move(b_)), solver_function};
}

// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include "cgpoisson_problem.h"
#include "Poisson.h"
#include "cg.h"
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

namespace
{
void pack_fn(std::span<const T> in, std::span<const std::int32_t> idx,
             std::span<T> out)
{
  for (std::size_t i = 0; i < idx.size(); ++i)
    out[i] = in[idx[i]];
}

void unpack_fn(std::span<const T> in, std::span<const std::int32_t> idx,
               std::span<T> out, std::function<T(T, T)> op)
{
  for (std::size_t i = 0; i < idx.size(); ++i)
    out[idx[i]] = op(out[idx[i]], in[i]);
}
} // namespace

std::tuple<std::shared_ptr<la::Vector<T>>, std::shared_ptr<fem::Function<T>>,
           std::function<int(fem::Function<T>&, const la::Vector<T>&)>>
cgpoisson::problem(std::shared_ptr<mesh::Mesh<double>> mesh, int order,
                   std::string scatterer)
{
  common::Timer t0("ZZZ FunctionSpace");

  auto element = basix::create_element<double>(
      basix::element::family::P, basix::cell::type::tetrahedron, order,
      basix::element::lagrange_variant::gll_warped,
      basix::element::dpc_variant::unset, false);

  auto V = std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace(mesh, element, {}));

  t0.stop();

  common::Timer t1("ZZZ Assemble");

  common::Timer t2("ZZZ Create boundary conditions");
  // Define boundary condition
  auto u0 = std::make_shared<fem::Function<T>>(V);
  u0->x()->set(0);

  // Find facets with bc applied
  const int tdim = mesh->topology()->dim();
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
  const std::vector<std::int32_t> bdofs = fem::locate_dofs_topological(
      *V->mesh()->topology_mutable(), *V->dofmap(), tdim - 1, bc_facets);

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
                                                              *form_poisson_L.at(order - 1), {V}, {{"w0", f}, {"w1", g}}, {}, {}, {}));
  // auto a = std::make_shared<fem::Form<T>>(fem::create_form<T>(
  //     *form_poisson_a.at(order - 1), {V, V},
  //     std::vector<std::shared_ptr<const fem::Function<T>>>{}, {}, {}));

  auto un = std::make_shared<fem::Function<T>>(V);
  auto M = std::make_shared<fem::Form<T>>(fem::create_form<T>(
                                                              *form_poisson_M.at(order - 1), {V}, {{"w0", un}}, {{}}, {}, {}));

  // Create la::Vector
  la::Vector<T> b(L->function_spaces()[0]->dofmap()->index_map,
                  L->function_spaces()[0]->dofmap()->index_map_bs());
  b.set(0);
  common::Timer t5("ZZZ Assemble vector");
  const std::vector constants_L = fem::pack_constants(*L);
  auto coeffs_L = fem::allocate_coefficient_storage(*L);
  fem::pack_coefficients(*L, coeffs_L);
  fem::assemble_vector<T>(b.mutable_array(), *L, constants_L,
                          fem::make_coefficients_span(coeffs_L));

  // Apply lifting to account for Dirichlet boundary condition
  // b <- b - A * x_bc
  fem::set_bc<T, double>(un->x()->mutable_array(), {bc}, -1.0);
  fem::assemble_vector(b.mutable_array(), *M);

  // Communicate ghost values
  b.scatter_rev(std::plus<T>());

  // Set BC dofs to zero (effectively zeroes columns of A)
  fem::set_bc<T, double>(b.mutable_array(), {bc}, 0.0);
  b.scatter_fwd();

  // Pack coefficients and constants

  if (un->x()->array().size() != b.array().size())
    throw std::runtime_error("error");
  // Create Function to hold solution
  auto u = std::make_shared<fem::Function<T>>(V);

  std::function<int(fem::Function<T>&, const la::Vector<T>&)> solver_function
      = [M, un, bc, scatterer](fem::Function<T>& u, const la::Vector<T>& b)
  {
    const std::vector<T> constants;
    auto coeff = fem::allocate_coefficient_storage(*M);

    auto V = M->function_spaces()[0];
    auto idx_map = V->dofmap()->index_map;
    int bs = V->dofmap()->bs();
    common::Scatterer sct(*idx_map, bs);

    std::vector<T> local_buffer(sct.local_buffer_size(), 0);
    std::vector<T> remote_buffer(sct.remote_buffer_size(), 0);

    common::Scatterer<>::type type;
    if (scatterer == "neighbor")
      type = common::Scatterer<>::type::neighbor;
    if (scatterer == "p2p")
      type = common::Scatterer<>::type::p2p;

    std::vector<MPI_Request> request = sct.create_request_vector(type);

    // Create function for computing the action of A on x (y = Ax)
    auto action = [&](la::Vector<T>& x, la::Vector<T>& y)
    {
      // Zero y
      y.set(0.0);

      // Update coefficient un (just copy data from x to un)
      std::copy(x.array().begin(), x.array().end(),
                un->x()->mutable_array().begin());

      // Compute action of A on x
      fem::pack_coefficients(*M, coeff);
      fem::assemble_vector(y.mutable_array(), *M, std::span<const T>(constants),
                           fem::make_coefficients_span(coeff));

      // Set BC dofs to zero (effectively zeroes rows of A)
      fem::set_bc<T, double>(y.mutable_array(), {bc}, 0.0);

      // Accumuate ghost values
      // y.scatter_rev(std::plus<T>());

      const std::int32_t local_size = bs * idx_map->size_local();
      const std::int32_t num_ghosts = bs * idx_map->num_ghosts();
      std::span<T> remote_data(y.mutable_array().data() + local_size,
                               num_ghosts);
      std::span<T> local_data(y.mutable_array().data(), local_size);
      sct.scatter_rev_begin<T>(remote_data, remote_buffer, local_buffer,
                               pack_fn, request, type);
      sct.scatter_rev_end<T>(local_buffer, local_data, unpack_fn,
                             std::plus<T>(), request);

      // Update ghost values
      sct.scatter_fwd_begin<T>(local_data, local_buffer, remote_buffer, pack_fn,
                               request, type);
      sct.scatter_fwd_end<T>(remote_buffer, remote_data, unpack_fn, request);
    };

    int num_it = linalg::cg(*u.x(), b, action, 100, 1e-6);
    return num_it;
  };

  return {std::make_shared<la::Vector<T>>(std::move(b)), u, solver_function};
}

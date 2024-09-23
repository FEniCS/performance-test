// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include "poisson_problem.h"
#include "Poisson.h"
#include <cfloat>
#include <cmath>
#include <dolfinx/common/Timer.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <memory>
#include <petscsys.h>
#include <utility>

using namespace dolfinx;
using T = PetscScalar;

std::tuple<std::shared_ptr<la::Vector<T>>, std::shared_ptr<fem::Function<T>>,
           std::function<int(fem::Function<T>&, const la::Vector<T>&)>>
poisson::problem(std::shared_ptr<mesh::Mesh<double>> mesh, int order)
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

  // Define variational forms
  auto L = std::make_shared<fem::Form<T>>(fem::create_form<T>(
                                                              *form_poisson_L.at(order - 1), {V}, {{"w0", f}, {"w1", g}}, {}, {}, {}));
  auto a = std::make_shared<fem::Form<T>>(
                                          fem::create_form<T>(*form_poisson_a.at(order - 1), {V, V}, {}, {}, {}, {}));

  // Create matrices and vector, and assemble system
  std::shared_ptr<la::petsc::Matrix> A = std::make_shared<la::petsc::Matrix>(
      fem::petsc::create_matrix(*a), false);

  common::Timer t4("ZZZ Assemble matrix");
  const std::vector constants_a = fem::pack_constants(*a);
  auto coeffs_a = fem::allocate_coefficient_storage(*a);
  fem::pack_coefficients(*a, coeffs_a);
  fem::assemble_matrix<T>(la::petsc::Matrix::set_block_fn(A->mat(), ADD_VALUES),
                          *a, constants_a,
                          fem::make_coefficients_span(coeffs_a), {bc});
  MatAssemblyBegin(A->mat(), MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(A->mat(), MAT_FLUSH_ASSEMBLY);
  fem::set_diagonal<T>(la::petsc::Matrix::set_fn(A->mat(), INSERT_VALUES), *V,
                       {bc});
  MatAssemblyBegin(A->mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A->mat(), MAT_FINAL_ASSEMBLY);
  t4.stop();

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
  fem::apply_lifting<T, double>(b.mutable_array(), {a}, {constants_L},
                                {fem::make_coefficients_span(coeffs_L)}, {{bc}},
                                {}, 1.0);
  b.scatter_rev(std::plus<>());
  fem::set_bc<T, double>(b.mutable_array(), {bc});
  t5.stop();

  t1.stop();

  // Create Function to hold solution
  auto u = std::make_shared<fem::Function<T>>(V);
  std::function<int(fem::Function<T>&, const la::Vector<T>&)> solver_function
      = [A](fem::Function<T>& u, const la::Vector<T>& b)
  {
    // Create solver
    la::petsc::KrylovSolver solver(MPI_COMM_WORLD);
    solver.set_from_options();
    solver.set_operator(A->mat());

    // Wrap la::Vector
    la::petsc::Vector _b(la::petsc::create_vector_wrap(b), false);
    la::petsc::Vector x(la::petsc::create_vector_wrap(*u.x()), false);

    // Solve
    int num_iter = solver.solve(x.vec(), _b.vec());
    return num_iter;
  };

  return {std::make_shared<la::Vector<T>>(std::move(b)), u, solver_function};
}

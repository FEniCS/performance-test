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
#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;

std::tuple<std::shared_ptr<la::Vector<PetscScalar>>,
           std::shared_ptr<fem::Function<PetscScalar>>,
           std::function<int(fem::Function<PetscScalar>&,
                             const la::Vector<PetscScalar>&)>>
poisson::problem(std::shared_ptr<mesh::Mesh> mesh, int order)
{
  common::Timer t0("ZZZ FunctionSpace");

  std::vector fs_poisson_a
      = {functionspace_form_Poisson_a1, functionspace_form_Poisson_a2,
         functionspace_form_Poisson_a3};

  auto V = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(*fs_poisson_a.at(order - 1), "v_0", mesh));

  t0.stop();

  common::Timer t1("ZZZ Assemble");

  std::shared_ptr<fem::DirichletBC<PetscScalar>> bc;
  {
    common::Timer t2("ZZZ Create boundary conditions");

    // Define boundary condition
    auto u0 = std::make_shared<fem::Function<PetscScalar>>(V);
    u0->x()->set(0);

    // Find facets with bc applied
    const int tdim = mesh->topology().dim();
    const std::vector<std::int32_t> bc_facets = mesh::locate_entities(
        *mesh, tdim - 1,
        [](const xt::xtensor<double, 2>& x) -> xt::xtensor<bool, 1>
        {
          auto x0 = xt::row(x, 0);
          return xt::isclose(x0, 0.0) or xt::isclose(x0, 1.0);
        });

    // Find constrained dofs
    const std::vector<std::int32_t> bdofs
        = fem::locate_dofs_topological(*V, tdim - 1, bc_facets);

    bc = std::make_shared<fem::DirichletBC<PetscScalar>>(u0, bdofs);
  }

  // Define coefficients
  common::Timer t3("ZZZ Create RHS function");
  auto f = std::make_shared<fem::Function<PetscScalar>>(V);
  auto g = std::make_shared<fem::Function<PetscScalar>>(V);
  f->interpolate(
      [](const xt::xtensor<double, 2>& x) -> xt::xarray<PetscScalar>
      {
        auto dx
            = xt::square(xt::row(x, 0) - 0.5) + xt::square(xt::row(x, 1) - 0.5);
        return 10 * xt::exp(-(dx) / 0.02);
      });

  g->interpolate([](const xt::xtensor<double, 2>& x) -> xt::xarray<PetscScalar>
                 { return xt::sin(5.0 * xt::row(x, 0)); });
  t3.stop();

  std::vector form_poisson_L
      = {form_Poisson_L1, form_Poisson_L2, form_Poisson_L3};
  std::vector form_poisson_a
      = {form_Poisson_a1, form_Poisson_a2, form_Poisson_a3};

  // Define variational forms
  auto L
      = std::make_shared<fem::Form<PetscScalar>>(fem::create_form<PetscScalar>(
          *form_poisson_L.at(order - 1), {V}, {{"w0", f}, {"w1", g}}, {}, {}));
  auto a
      = std::make_shared<fem::Form<PetscScalar>>(fem::create_form<PetscScalar>(
          *form_poisson_a.at(order - 1), {V, V},
          std::vector<std::shared_ptr<const fem::Function<PetscScalar>>>{}, {},
          {}));

  // Create matrices and vector, and assemble system
  std::shared_ptr<la::petsc::Matrix> A = std::make_shared<la::petsc::Matrix>(
      fem::petsc::create_matrix(*a), false);

  common::Timer t4("ZZZ Assemble matrix");
  const std::vector constants_a = fem::pack_constants(*a);
  auto coeffs_a = fem::allocate_coefficient_storage(*a);
  fem::pack_coefficients(*a, coeffs_a);
  fem::assemble_matrix<PetscScalar>(
      la::petsc::Matrix::set_block_fn(A->mat(), ADD_VALUES), *a,
      tcb::make_span(constants_a), fem::make_coefficients_span(coeffs_a), {bc});
  MatAssemblyBegin(A->mat(), MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(A->mat(), MAT_FLUSH_ASSEMBLY);
  fem::set_diagonal<PetscScalar>(
      la::petsc::Matrix::set_fn(A->mat(), INSERT_VALUES), *V, {bc});
  MatAssemblyBegin(A->mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A->mat(), MAT_FINAL_ASSEMBLY);
  t4.stop();

  // Create la::Vector
  la::Vector<PetscScalar> b(L->function_spaces()[0]->dofmap()->index_map,
                            L->function_spaces()[0]->dofmap()->index_map_bs());
  b.set(0);
  common::Timer t5("ZZZ Assemble vector");
  const std::vector constants_L = fem::pack_constants(*L);
  auto coeffs_L = fem::allocate_coefficient_storage(*L);
  fem::pack_coefficients(*L, coeffs_L);
  fem::assemble_vector<PetscScalar>(b.mutable_array(), *L, constants_L,
                                    fem::make_coefficients_span(coeffs_L));
  fem::apply_lifting(b.mutable_array(), {a}, {constants_L},
                     {fem::make_coefficients_span(coeffs_L)}, {{bc}}, {}, 1.0);
  b.scatter_rev(common::IndexMap::Mode::add);
  fem::set_bc(b.mutable_array(), {bc});
  t5.stop();

  t1.stop();

  // Create Function to hold solution
  auto u = std::make_shared<fem::Function<PetscScalar>>(V);
  std::function<int(fem::Function<PetscScalar>&,
                    const la::Vector<PetscScalar>&)>
      solver_function
      = [A](fem::Function<PetscScalar>& u, const la::Vector<PetscScalar>& b)
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

  return {std::make_shared<la::Vector<PetscScalar>>(std::move(b)), u,
          solver_function};
}

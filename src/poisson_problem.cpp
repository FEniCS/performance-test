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
#include <dolfinx/la/PETScKrylovSolver.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <utility>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

std::tuple<std::shared_ptr<dolfinx::la::Vector<PetscScalar>>,
           std::shared_ptr<dolfinx::fem::Function<PetscScalar>>,
           std::function<int(dolfinx::fem::Function<PetscScalar>&,
                             const dolfinx::la::Vector<PetscScalar>&)>>
poisson::problem(std::shared_ptr<dolfinx::mesh::Mesh> mesh, int order)
{
  dolfinx::common::Timer t0("ZZZ FunctionSpace");

  std::vector fs_poisson_a
      = {functionspace_form_Poisson_a1, functionspace_form_Poisson_a2,
         functionspace_form_Poisson_a3};

  auto V = dolfinx::fem::create_functionspace(*fs_poisson_a.at(order - 1),
                                              "v_0", mesh);

  t0.stop();

  dolfinx::common::Timer t1("ZZZ Assemble");

  dolfinx::common::Timer t2("ZZZ Create boundary conditions");
  // Define boundary condition
  auto u0 = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  std::fill(u0->x()->mutable_array().begin(), u0->x()->mutable_array().end(),
            0.0);

  // Find facets with bc applied
  const int tdim = mesh->topology().dim();
  const std::vector<std::int32_t> bc_facets = dolfinx::mesh::locate_entities(
      *mesh, tdim - 1,
      [](const xt::xtensor<double, 2>& x) -> xt::xtensor<bool, 1>
      {
        auto x0 = xt::row(x, 0);
        return xt::isclose(x0, 0.0) or xt::isclose(x0, 1.0);
      });

  // Find constrained dofs
  const std::vector<std::int32_t> bdofs
      = dolfinx::fem::locate_dofs_topological(*V, tdim - 1, bc_facets);

  auto bc = std::make_shared<dolfinx::fem::DirichletBC<PetscScalar>>(u0, bdofs);
  t2.stop();

  // Define coefficients
  dolfinx::common::Timer t3("ZZZ Create RHS function");
  auto f = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  auto g = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  f->interpolate(
      [](const xt::xtensor<double, 2>& x) -> xt::xarray<PetscScalar>
      {
        auto dx
            = xt::square(xt::row(x, 0) - 0.5) + xt::square(xt::row(x, 1) - 0.5);
        return 10 * xt::exp(-(dx) / 0.02);
      });

  g->interpolate(
      [](const xt::xtensor<double, 2>& x) -> xt::xarray<PetscScalar> {
        return xt::sin(5.0 * xt::row(x, 0));
      });
  t3.stop();

  std::vector form_poisson_L
      = {form_Poisson_L1, form_Poisson_L2, form_Poisson_L3};
  std::vector form_poisson_a
      = {form_Poisson_a1, form_Poisson_a2, form_Poisson_a3};

  // Define variational forms
  auto L = std::make_shared<dolfinx::fem::Form<PetscScalar>>(
      dolfinx::fem::create_form<PetscScalar>(*form_poisson_L.at(order - 1), {V},
                                             {{"w0", f}, {"w1", g}}, {}, {}));
  auto a = std::make_shared<
      dolfinx::fem::Form<PetscScalar>>(dolfinx::fem::create_form<PetscScalar>(
      *form_poisson_a.at(order - 1), {V, V},
      std::vector<std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>>{},
      {}, {}));

  // Create matrices and vector, and assemble system
  std::shared_ptr<dolfinx::la::PETScMatrix> A
      = std::make_shared<dolfinx::la::PETScMatrix>(
          dolfinx::fem::create_matrix(*a), false);

  // Wrap la::Vector with Petsc Vec
  dolfinx::la::Vector<PetscScalar> bx(
      L->function_spaces()[0]->dofmap()->index_map,
      L->function_spaces()[0]->dofmap()->index_map_bs());
  Vec b_vec = dolfinx::la::create_ghosted_vector(
      *(bx.map()), bx.bs(), tcb::span<PetscScalar>(bx.mutable_array()));
  dolfinx::la::PETScVector b(b_vec, false);

  dolfinx::common::Timer t4("ZZZ Assemble matrix");
  const std::vector constants_a = dolfinx::fem::pack_constants(*a);
  const auto coeffs_a = dolfinx::fem::pack_coefficients(*a);
  dolfinx::fem::assemble_matrix(
      dolfinx::la::PETScMatrix::set_block_fn(A->mat(), ADD_VALUES), *a,
      tcb::make_span(constants_a), {coeffs_a.first, coeffs_a.second}, {bc});
  MatAssemblyBegin(A->mat(), MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(A->mat(), MAT_FLUSH_ASSEMBLY);
  dolfinx::fem::set_diagonal(
      dolfinx::la::PETScMatrix::set_fn(A->mat(), INSERT_VALUES), *V, {bc});
  MatAssemblyBegin(A->mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A->mat(), MAT_FINAL_ASSEMBLY);
  t4.stop();

  // Zero PETSc vector
  Vec b_local;
  VecGhostGetLocalForm(b.vec(), &b_local);
  PetscInt n = 0;
  VecGetSize(b_local, &n);
  PetscScalar* array = nullptr;
  VecGetArray(b_local, &array);
  VecSet(b_local, 0.0);
  VecRestoreArray(b_local, &array);
  VecGhostRestoreLocalForm(b.vec(), &b_local);

  dolfinx::common::Timer t5("ZZZ Assemble vector");
  const std::vector constants_L = dolfinx::fem::pack_constants(*L);
  const auto coeffs_L = dolfinx::fem::pack_coefficients(*L);
  dolfinx::fem::assemble_vector_petsc(b.vec(), *L, constants_L,
                                      {coeffs_L.first, coeffs_L.second});
  dolfinx::fem::apply_lifting_petsc(b.vec(), {a}, {constants_L},
                                    {{coeffs_L.first, coeffs_L.second}}, {{bc}},
                                    {}, 1.0);
  VecGhostUpdateBegin(b.vec(), ADD_VALUES, SCATTER_REVERSE);
  VecGhostUpdateEnd(b.vec(), ADD_VALUES, SCATTER_REVERSE);
  dolfinx::fem::set_bc_petsc(b.vec(), {bc}, nullptr);
  t5.stop();

  t1.stop();

  // Create Function to hold solution
  auto u = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);

  std::function<int(dolfinx::fem::Function<PetscScalar>&,
                    const dolfinx::la::Vector<PetscScalar>&)>
      solver_function = [A](dolfinx::fem::Function<PetscScalar>& u,
                            const dolfinx::la::Vector<PetscScalar>& b)
  {
    // Create solver
    dolfinx::la::PETScKrylovSolver solver(MPI_COMM_WORLD);
    solver.set_from_options();
    solver.set_operator(A->mat());

    // Wrap dolfinx::la::Vector
    dolfinx::la::Vector<PetscScalar>& bnc
        = const_cast<dolfinx::la::Vector<PetscScalar>&>(b);
    Vec b_petsc = dolfinx::la::create_ghosted_vector(
        *(b.map()), b.bs(), tcb::span<PetscScalar>(bnc.mutable_array()));

    // Solve
    int num_iter = solver.solve(u.vector(), b_petsc);
    return num_iter;
  };
  return {std::make_shared<dolfinx::la::Vector<PetscScalar>>(std::move(bx)), u,
          solver_function};
}

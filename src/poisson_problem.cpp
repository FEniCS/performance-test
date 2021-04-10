// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include "poisson_problem.h"
#include "Poisson.h"
#include <Eigen/Dense>
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

std::tuple<dolfinx::la::Vector<PetscScalar>,
           std::shared_ptr<dolfinx::fem::Function<PetscScalar>>,
           std::function<int(dolfinx::fem::Function<PetscScalar>&,
                             const dolfinx::la::Vector<PetscScalar>&)>>
poisson::problem(std::shared_ptr<dolfinx::mesh::Mesh> mesh)
{
  dolfinx::common::Timer t0("ZZZ FunctionSpace");

  auto V = dolfinx::fem::create_functionspace(
      create_functionspace_form_Poisson_a, "u", mesh);

  t0.stop();

  dolfinx::common::Timer t1("ZZZ Assemble");

  dolfinx::common::Timer t2("ZZZ Create boundary conditions");
  // Define boundary condition
  auto u0 = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  std::fill(u0->x()->mutable_array().begin(), u0->x()->mutable_array().end(),
            0.0);

  const std::vector<std::int32_t> bdofs = dolfinx::fem::locate_dofs_geometrical(
      {*V}, [](const xt::xtensor<double, 2>& x) -> xt::xtensor<bool, 1> {
        auto x0 = xt::row(x, 0);
        return xt::isclose(x0, 0.0) or xt::isclose(x0, 1.0);
      });

  auto bc = std::make_shared<dolfinx::fem::DirichletBC<PetscScalar>>(u0, bdofs);
  t2.stop();

  // Define coefficients
  dolfinx::common::Timer t3("ZZZ Create RHS function");
  auto f = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  auto g = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  f->interpolate(
      [](const xt::xtensor<double, 2>& x) -> xt::xarray<PetscScalar> {
        auto dx
            = xt::square(xt::row(x, 0) - 0.5) + xt::square(xt::row(x, 1) - 0.5);
        return 10 * xt::exp(-(dx) / 0.02);
      });

  g->interpolate(
      [](const xt::xtensor<double, 2>& x) -> xt::xarray<PetscScalar> {
        return xt::sin(5.0 * xt::row(x, 0));
      });
  t3.stop();

  // Define variational forms
  auto L = dolfinx::fem::create_form<PetscScalar>(create_form_Poisson_L, {V},
                                                  {{"f", f}, {"g", g}}, {}, {});
  auto a = dolfinx::fem::create_form<PetscScalar>(create_form_Poisson_a, {V, V},
                                                  {}, {}, {});

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

  MatZeroEntries(A->mat());
  dolfinx::common::Timer t4("ZZZ Assemble matrix");
  dolfinx::fem::assemble_matrix(dolfinx::la::PETScMatrix::add_fn(A->mat()), *a,
                                {bc});
  dolfinx::fem::add_diagonal(dolfinx::la::PETScMatrix::add_fn(A->mat()), *V,
                             {bc});
  MatAssemblyBegin(A->mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A->mat(), MAT_FINAL_ASSEMBLY);
  t4.stop();

  VecSet(b.vec(), 0.0);
  VecGhostUpdateBegin(b.vec(), INSERT_VALUES, SCATTER_FORWARD);
  VecGhostUpdateEnd(b.vec(), INSERT_VALUES, SCATTER_FORWARD);

  dolfinx::common::Timer t5("ZZZ Assemble vector");
  dolfinx::fem::assemble_vector_petsc(b.vec(), *L);
  dolfinx::fem::apply_lifting_petsc(b.vec(), {a}, {{bc}}, {}, 1.0);
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
                            const dolfinx::la::Vector<PetscScalar>& b) {
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

  return {std::move(bx), u, solver_function};
}

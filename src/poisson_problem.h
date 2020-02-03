// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "Poisson.h"
#include <Eigen/Dense>
#include <cfloat>
#include <dolfinx/common/Timer.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <utility>

namespace poisson
{

std::tuple<dolfinx::la::PETScMatrix, dolfinx::la::PETScVector,
           std::shared_ptr<dolfinx::function::Function>>
problem(std::shared_ptr<dolfinx::mesh::Mesh> mesh)
{
  dolfinx::common::Timer t0("ZZZ FunctionSpace");

  std::shared_ptr<dolfinx::function::FunctionSpace> V
      = dolfinx::fem::create_functionspace(Poisson_functionspace_create, mesh);

  t0.stop();

  dolfinx::common::Timer t1("ZZZ Assemble");

  // Define variational forms
  std::shared_ptr<dolfinx::fem::Form> form_L
      = dolfinx::fem::create_form(Poisson_linearform_create, {V});

  std::shared_ptr<dolfinx::fem::Form> form_a
      = dolfinx::fem::create_form(Poisson_bilinearform_create, {V, V});

  // Define variational forms
  ufc_form* linear_form = Poisson_linearform_create();
  auto L = std::make_shared<dolfinx::fem::Form>(
      dolfinx::fem::create_form(*linear_form, {V}));
  std::free(linear_form);

  ufc_form* bilinear_form = Poisson_bilinearform_create();
  auto a = std::make_shared<dolfinx::fem::Form>(
      dolfinx::fem::create_form(*bilinear_form, {V, V}));
  std::free(bilinear_form);

  // Attach 'coordinate mapping' to mesh
  auto cmap = a->coordinate_mapping();
  mesh->geometry().coord_mapping = cmap;

  // Define boundary condition
  auto u0 = std::make_shared<dolfinx::function::Function>(V);
  VecSet(u0->vector().vec(), 0.0);
  VecGhostUpdateBegin(u0->vector().vec(), INSERT_VALUES, SCATTER_FORWARD);
  VecGhostUpdateEnd(u0->vector().vec(), INSERT_VALUES, SCATTER_FORWARD);

  const Eigen::Array<PetscInt, Eigen::Dynamic, 1> bdofs
      = dolfinx::fem::locate_dofs_geometrical(*V, [](auto& x) {
          return (x.row(0) < DBL_EPSILON or x.row(0) > 1.0 - DBL_EPSILON);
        });

  auto bc = std::make_shared<dolfinx::fem::DirichletBC>(u0, bdofs);

  // Define coefficients
  auto f = std::make_shared<dolfinx::function::Function>(V);
  auto g = std::make_shared<dolfinx::function::Function>(V);
  f->interpolate([](auto& x) {
    auto dx = x.row(0) - 0.5;
    auto dy = x.row(1) - 0.5;
    return 10 * (-(dx * dx + dy * dy).exp() / 0.02);
  });
  g->interpolate([](auto& x) {
    {
      return (5.0 * x.row(0)).sin();
    }
  });

  L->set_coefficients({{"f", f}, {"g", g}});

  // Create matrices and vector, and assemble system
  dolfinx::la::PETScMatrix A = dolfinx::fem::create_matrix(*a);
  dolfinx::la::PETScVector b(*L->function_space(0)->dofmap()->index_map);

  MatZeroEntries(A.mat());
  dolfinx::common::Timer t2("ZZZ Assemble matrix");
  dolfinx::fem::assemble_matrix(A.mat(), *a, {bc});
  dolfinx::fem::add_diagonal(A.mat(), *V, {bc});
  MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);
  t2.stop();

  VecSet(b.vec(), 0.0);
  VecGhostUpdateBegin(b.vec(), INSERT_VALUES, SCATTER_FORWARD);
  VecGhostUpdateEnd(b.vec(), INSERT_VALUES, SCATTER_FORWARD);

  dolfinx::common::Timer t3("ZZZ Assemble vector");
  dolfinx::fem::assemble_vector(b.vec(), *L);
  dolfinx::fem::apply_lifting(b.vec(), {a}, {{bc}}, {}, 1.0);
  VecGhostUpdateBegin(b.vec(), ADD_VALUES, SCATTER_REVERSE);
  VecGhostUpdateEnd(b.vec(), ADD_VALUES, SCATTER_REVERSE);
  dolfinx::fem::set_bc(b.vec(), {bc}, nullptr);
  t3.stop();

  t1.stop();

  // Create Function to hold solution
  auto u = std::make_shared<dolfinx::function::Function>(V);

  return std::tuple<dolfinx::la::PETScMatrix, dolfinx::la::PETScVector,
                    std::shared_ptr<dolfinx::function::Function>>(
      std::move(A), std::move(b), u);
}
} // namespace poisson

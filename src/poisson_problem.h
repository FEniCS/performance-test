// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "Poisson.h"
#include <Eigen/Dense>
#include <cfloat>
#include <dolfin/common/Timer.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/mesh/Mesh.h>
#include <memory>
#include <utility>

namespace poisson
{

std::tuple<dolfin::la::PETScMatrix, dolfin::la::PETScVector,
           std::shared_ptr<dolfin::function::Function>>
problem(std::shared_ptr<dolfin::mesh::Mesh> mesh)
{
  dolfin::common::Timer t0("ZZZ FunctionSpace");

  std::shared_ptr<dolfin::function::FunctionSpace> V
      = dolfin::fem::create_functionspace(Poisson_functionspace_create, mesh);

  t0.stop();

  dolfin::common::Timer t1("ZZZ Assemble");

  // Define boundary condition
  auto u0 = std::make_shared<dolfin::function::Function>(V);
  VecSet(u0->vector().vec(), 0.0);

  auto bc = std::make_shared<dolfin::fem::DirichletBC>(V, u0, [](auto x) {
    return (x.col(0) < DBL_EPSILON or x.col(0) > 1.0 - DBL_EPSILON);
  });

  // Define variational forms
  auto form_L = std::unique_ptr<ufc_form, decltype(free)*>(
      Poisson_linearform_create(), free);
  auto form_a = std::unique_ptr<ufc_form, decltype(free)*>(
      Poisson_bilinearform_create(), free);

  // Define variational forms
  ufc_form* linear_form = Poisson_linearform_create();
  auto L = std::make_shared<dolfin::fem::Form>(
      dolfin::fem::create_form(*linear_form, {V}));
  std::free(linear_form);

  ufc_form* bilinear_form = Poisson_bilinearform_create();
  auto a = std::make_shared<dolfin::fem::Form>(
      dolfin::fem::create_form(*bilinear_form, {V, V}));
  std::free(bilinear_form);

  // Attach 'coordinate mapping' to mesh
  auto cmap = a->coordinate_mapping();
  mesh->geometry().coord_mapping = cmap;

  auto f = std::make_shared<dolfin::function::Function>(V);
  auto g = std::make_shared<dolfin::function::Function>(V);
  f->interpolate([](auto x) {
    auto dx = x.col(0) - 0.5;
    auto dy = x.col(1) - 0.5;
    return 10 * (-(dx * dx + dy * dy).exp() / 0.02);
  });
  g->interpolate([](auto x) {
    {
      return (5.0 * x.col(0)).sin();
    }
  });

  L->set_coefficients({{"f", f}, {"g", g}});

  // Create matrices and vector, and assemble system
  dolfin::la::PETScMatrix A = dolfin::fem::create_matrix(*a);
  dolfin::la::PETScVector b(*L->function_space(0)->dofmap()->index_map);

  MatZeroEntries(A.mat());
  dolfin::common::Timer t2("ZZZ Assemble matrix");
  dolfin::fem::assemble_matrix(A.mat(), *a, {bc});
  MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);
  t2.stop();

  VecSet(b.vec(), 0.0);
  VecGhostUpdateBegin(b.vec(), INSERT_VALUES, SCATTER_FORWARD);
  VecGhostUpdateEnd(b.vec(), INSERT_VALUES, SCATTER_FORWARD);

  dolfin::common::Timer t3("ZZZ Assemble vector");
  dolfin::fem::assemble_vector(b.vec(), *L);
  dolfin::fem::apply_lifting(b.vec(), {a}, {{bc}}, {}, 1.0);
  VecGhostUpdateBegin(b.vec(), ADD_VALUES, SCATTER_REVERSE);
  VecGhostUpdateEnd(b.vec(), ADD_VALUES, SCATTER_REVERSE);
  dolfin::fem::set_bc(b.vec(), {bc}, nullptr);
  t3.stop();

  t1.stop();

  // Create Function to hold solution
  auto u = std::make_shared<dolfin::function::Function>(V);

  return std::tuple<dolfin::la::PETScMatrix, dolfin::la::PETScVector,
                    std::shared_ptr<dolfin::function::Function>>(
      std::move(A), std::move(b), u);
}
} // namespace poisson

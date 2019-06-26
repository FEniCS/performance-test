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

  // auto V = std::make_shared<Poisson::FunctionSpace>(mesh);
  ufc_function_space* space = Poisson_functionspace_create();
  ufc_dofmap* ufc_map = space->create_dofmap();
  ufc_finite_element* ufc_element = space->create_element();
  auto V = std::make_shared<dolfin::function::FunctionSpace>(
      mesh, std::make_shared<dolfin::fem::FiniteElement>(*ufc_element),
      std::make_shared<dolfin::fem::DofMap>(*ufc_map, *mesh));
  std::free(ufc_element);
  std::free(ufc_map);
  std::free(space);

  t0.stop();

  dolfin::common::Timer t1("ZZZ Assemble");

  // Define boundary condition
  auto u0 = std::make_shared<dolfin::function::Function>(V);
  VecSet(u0->vector().vec(), 0.0);

  auto bc = std::make_shared<dolfin::fem::DirichletBC>(
      V, u0, [](auto x, bool only_boundary) {
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
      *linear_form,
      std::initializer_list<
          std::shared_ptr<const dolfin::function::FunctionSpace>>{V});
  std::free(linear_form);

  ufc_form* bilinear_form = Poisson_bilinearform_create();
  auto a = std::make_shared<dolfin::fem::Form>(
      *bilinear_form,
      std::initializer_list<
          std::shared_ptr<const dolfin::function::FunctionSpace>>{V, V});
  std::free(bilinear_form);

  // Attach 'coordinate mapping' to mesh
  auto cmap = a->coordinate_mapping();
  mesh->geometry().coord_mapping = cmap;

  auto f = std::make_shared<dolfin::function::Function>(V);
  auto g = std::make_shared<dolfin::function::Function>(V);
  f->interpolate([](auto values, auto x) {
    for (Eigen::Index i = 0; i < x.rows(); ++i)
    {
      double dx = x(i, 0) - 0.5;
      double dy = x(i, 1) - 0.5;
      values(i, 0) = 10 * exp(-(dx * dx + dy * dy) / 0.02);
    }
  });
  g->interpolate([](auto values, auto x) {
    {
      for (Eigen::Index i = 0; i < x.rows(); ++i)
        values(i, 0) = sin(5.0 * x(i, 0));
    }
  });

  L->set_coefficients({{"f", f}, {"g", g}});

  // Create matrices and vector, and assemble system
  dolfin::la::PETScMatrix A = dolfin::fem::create_matrix(*a);
  dolfin::la::PETScVector b(*L->function_space(0)->dofmap()->index_map());

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

// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#pragma once

#include "Elasticity.h"
#include <Eigen/Dense>
#include <dolfin/common/Timer.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/DofMap.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/assembler.h>
#include <dolfin/fem/utils.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/VectorSpaceBasis.h>
#include <dolfin/la/utils.h>
#include <dolfin/mesh/Geometry.h>
#include <dolfin/mesh/Mesh.h>
#include <memory>
#include <utility>

namespace elastic
{
// Function to compute the near nullspace for elasticity - it is made up
// of the six rigid body modes

dolfin::la::VectorSpaceBasis
build_near_nullspace(const dolfin::function::FunctionSpace& V)
{
  // Get subspaces
  auto V0 = V.sub({0});
  auto V1 = V.sub({1});
  auto V2 = V.sub({2});

  // Create vectors for nullspace basis
  std::vector<std::shared_ptr<dolfin::la::PETScVector>> basis_vec;
  for (std::size_t i = 0; i < 6; ++i)
  {
    basis_vec.push_back(
        std::make_shared<dolfin::la::PETScVector>(*V.dofmap()->index_map));
  }

  {
    // Unwrap the PETSc Vec objects to allow array (Eigen) access
    std::vector<dolfin::la::VecWrapper> basis;
    for (auto vec : basis_vec)
      basis.push_back(dolfin::la::VecWrapper(vec->vec()));

    // x0, x1, x2 translations
    V0->dofmap()->set(basis[0].x, 1.0);
    V1->dofmap()->set(basis[1].x, 1.0);
    V2->dofmap()->set(basis[2].x, 1.0);

    // Rotations
    V0->set_x(basis[3].x, -1.0, 1);
    V1->set_x(basis[3].x, 1.0, 0);

    V0->set_x(basis[4].x, 1.0, 2);
    V2->set_x(basis[4].x, -1.0, 0);

    V2->set_x(basis[5].x, 1.0, 1);
    V1->set_x(basis[5].x, -1.0, 2);
  }

  // Create vector space and orthonormalize
  dolfin::la::VectorSpaceBasis vector_space(basis_vec);
  vector_space.orthonormalize();
  return vector_space;
}

std::tuple<dolfin::la::PETScMatrix, dolfin::la::PETScVector,
           std::shared_ptr<dolfin::function::Function>>
problem(std::shared_ptr<dolfin::mesh::Mesh> mesh)
{
  dolfin::common::Timer t0("ZZZ FunctionSpace");

  std::shared_ptr<dolfin::function::FunctionSpace> V
      = dolfin::fem::create_functionspace(Elasticity_functionspace_create,
                                          mesh);

  t0.stop();

  dolfin::common::Timer t1("ZZZ Assemble prep");

  // Define boundary condition
  auto u0 = std::make_shared<dolfin::function::Function>(V);
  VecSet(u0->vector().vec(), 0.0);

  // Bottom (x[1] = 0) surface
  auto bc = std::make_shared<dolfin::fem::DirichletBC>(
      V, u0, [](auto& x) { return x.row(1) < 1.0e-8; });

  // Define variational forms

  std::shared_ptr <dolfin::fem::Form> L =
      dolfin::fem::create_form(Elasticity_linearform_create, {V});

  std::shared_ptr<dolfin::fem::Form> a =
    dolfin::fem::create_form(Elasticity_bilinearform_create,{V, V});


  // Attach 'coordinate mapping' to mesh
  auto cmap = a->coordinate_mapping();
  mesh->geometry().coord_mapping = cmap;

  auto f = std::make_shared<dolfin::function::Function>(V);
  f->interpolate([](auto& x) {
    auto dx = x.row(0) - 0.5;
    auto dz = x.row(2) - 0.5;
    auto r = dx * dx + dz * dz;
    Eigen::Array<double, 3, Eigen::Dynamic, Eigen::RowMajor> values(3,
                                                                    x.cols());
    values.row(0) = -dz * r.sqrt() * x.row(1);
    values.row(1) = 1.0;
    values.row(2) = dx * r.sqrt() * x.row(1);
    return values;
  });

  L->set_coefficients({{"f", f}});

  t1.stop();

  // Create matrices and vector, and assemble system
  dolfin::la::PETScMatrix A = dolfin::fem::create_matrix(*a);
  dolfin::la::PETScVector b(*L->function_space(0)->dofmap()->index_map);

  MatZeroEntries(A.mat());

  dolfin::common::Timer t2("ZZZ Assemble matrix");
  dolfin::fem::assemble_matrix(A.mat(), *a, {bc});
  dolfin::fem::add_diagonal(A.mat(), *V, {bc});
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

  dolfin::common::Timer t4("ZZZ Create near-nullspace");

  // Create Function to hold solution
  auto u = std::make_shared<dolfin::function::Function>(V);

  // Build near-nullspace and attach to matrix
  dolfin::la::VectorSpaceBasis nullspace = build_near_nullspace(*V);
  A.set_near_nullspace(nullspace);
  t4.stop();

  return std::tuple<dolfin::la::PETScMatrix, dolfin::la::PETScVector,
                    std::shared_ptr<dolfin::function::Function>>(
      std::move(A), std::move(b), u);
}
} // namespace elastic

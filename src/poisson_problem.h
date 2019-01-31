// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <Eigen/Dense>
#include <memory>
#include <utility>

#include <dolfin/common/Timer.h>
#include <dolfin/fem/SystemAssembler.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/SubDomain.h>

#include "Poisson.h"

namespace poisson
{
// Source term (right-hand side)
class Source : public dolfin::function::Expression
{
public:
  Source() : dolfin::function::Expression({}) {}

  void eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
                values,
            const Eigen::Ref<const dolfin::EigenRowArrayXXd> x,
            const dolfin::mesh::Cell& cell) const
  {
    for (Eigen::Index i = 0; i < x.rows(); ++i)
    {
      double dx = x(i, 0) - 0.5;
      double dy = x(i, 1) - 0.5;
      values(i, 0) = 10 * exp(-(dx * dx + dy * dy) / 0.02);
    }
  }
};

// Normal derivative (Neumann boundary condition)
class dUdN : public dolfin::function::Expression
{
public:
  dUdN() : dolfin::function::Expression({}) {}

  void eval(Eigen::Ref<Eigen::Array<PetscScalar, Eigen::Dynamic, Eigen::Dynamic,
                                    Eigen::RowMajor>>
                values,
            const Eigen::Ref<const dolfin::EigenRowArrayXXd> x,
            const dolfin::mesh::Cell& cell) const
  {
    for (Eigen::Index i = 0; i < x.rows(); ++i)
      values(i, 0) = sin(5.0 * x(i, 0));
  }
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public dolfin::mesh::SubDomain
{
public:
  dolfin::EigenArrayXb inside(Eigen::Ref<const dolfin::EigenRowArrayXXd> x,
                              bool on_boundary) const
  {
    dolfin::EigenArrayXb b(x.rows());
    for (Eigen::Index i = 0; i < x.rows(); ++i)
      b(i, 0) = x(i, 0) < DOLFIN_EPS or x(i, 0) > (1.0 - DOLFIN_EPS);
    return b;
  }
};

std::tuple<dolfin::la::PETScMatrix, dolfin::la::PETScVector,
           std::shared_ptr<dolfin::function::Function>>
problem(std::shared_ptr<dolfin::mesh::Mesh> mesh)
{
  dolfin::common::Timer t0("ZZZ FunctionSpace");

  // auto V = std::make_shared<Poisson::FunctionSpace>(mesh);
  auto space = std::unique_ptr<dolfin_function_space>(PoissonFunctionSpace());
  auto V = std::make_shared<dolfin::function::FunctionSpace>(
      mesh,
      std::make_shared<dolfin::fem::FiniteElement>(
          std::shared_ptr<ufc_finite_element>(space->element())),
      std::make_shared<dolfin::fem::DofMap>(
          std::shared_ptr<ufc_dofmap>(space->dofmap()), *mesh));
  t0.stop();

  dolfin::common::Timer t1("ZZZ Assemble");

  // Define boundary condition
  auto u0 = std::make_shared<dolfin::function::Function>(V);
  VecSet(u0->vector().vec(), 0.0);

  auto boundary = std::make_shared<DirichletBoundary>();
  auto bc = std::make_shared<dolfin::fem::DirichletBC>(V, u0, *boundary);

  // Define variational forms
  auto form_L = std::unique_ptr<dolfin_form>(PoissonLinearForm());
  auto form_a = std::unique_ptr<dolfin_form>(PoissonBilinearForm());

  // Define variational forms
  dolfin::fem::Form a(
      std::shared_ptr<ufc_form>(form_a->form()),
      std::initializer_list<
          std::shared_ptr<const dolfin::function::FunctionSpace>>{V, V});
  dolfin::fem::Form L(
      std::shared_ptr<ufc_form>(form_L->form()),
      std::initializer_list<
          std::shared_ptr<const dolfin::function::FunctionSpace>>{V});

  // Attach 'coordinate mapping' to mesh
  auto cmap = a.coordinate_mapping();
  mesh->geometry().coord_mapping = cmap;

  Source f_expr;
  dUdN g_expr;
  auto f = std::make_shared<dolfin::function::Function>(V);
  auto g = std::make_shared<dolfin::function::Function>(V);
  f->interpolate(f_expr);
  g->interpolate(g_expr);

  L.set_coefficient_index_to_name_map(form_L->coefficient_number_map);
  L.set_coefficient_name_to_index_map(form_L->coefficient_name_map);
  L.set_coefficients({{"f", f}, {"g", g}});

  std::shared_ptr<dolfin::fem::Form> _a(&a, [](dolfin::fem::Form* ptr) {});
  std::shared_ptr<dolfin::fem::Form> _L(&L, [](dolfin::fem::Form* ptr) {});

  // Create matrices and vector, and assemble system
  dolfin::la::PETScMatrix A(dolfin::fem::create_matrix(a));
  MatZeroEntries(A.mat());

  dolfin::common::Timer t2("ZZZ Assemble matrix");
  dolfin::fem::assemble_matrix(A.mat(), a, {bc});
  MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);
  t2.stop();

  dolfin::la::PETScVector b(*L.function_space(0)->dofmap()->index_map());
  VecSet(b.vec(), 0.0);
  VecGhostUpdateBegin(b.vec(), INSERT_VALUES, SCATTER_FORWARD);
  VecGhostUpdateEnd(b.vec(), INSERT_VALUES, SCATTER_FORWARD);

  dolfin::common::Timer t3("ZZZ Assemble vector");
  dolfin::fem::assemble_vector(b.vec(), L);
  dolfin::fem::apply_lifting(b.vec(), {_a}, {{bc}}, {}, 1.0);
  VecGhostUpdateBegin(b.vec(), ADD_VALUES, SCATTER_REVERSE);
  VecGhostUpdateEnd(b.vec(), ADD_VALUES, SCATTER_REVERSE);
  dolfin::fem::set_bc(b.vec(), {bc}, nullptr);
  t3.stop();

  // Create assembler
  // dolfin::fem::SystemAssembler assembler(_a, _L, {bc});
  // assembler.assemble(A, b);

  t1.stop();

  // Create Function to hold solution
  auto u = std::make_shared<dolfin::function::Function>(V);

  return std::tuple<dolfin::la::PETScMatrix, dolfin::la::PETScVector,
                    std::shared_ptr<dolfin::function::Function>>(
      std::move(A), std::move(b), u);
}
} // namespace poisson

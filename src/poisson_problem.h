// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
// Licensed under the MIT License. See LICENSE file in the project
// root for full license information.

#pragma once

#include <memory>
#include <utility>
#include <dolfin/common/Array.h>
#include <dolfin/common/Timer.h>
#include <dolfin/fem/SystemAssembler.h>
#include <dolfin/function/Constant.h>
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
  class Source : public dolfin::Expression
  {
  public:
    void eval(dolfin::Array<double>& values,
              const dolfin::Array<double>& x) const
    {
      double dx = x[0] - 0.5;
      double dy = x[1] - 0.5;
      values[0] = 10*exp(-(dx*dx + dy*dy)/0.02);
    }
  };

  // Normal derivative (Neumann boundary condition)
  class dUdN : public dolfin::Expression
  {
  public:
    void eval(dolfin::Array<double>& values,
              const dolfin::Array<double>& x) const
    { values[0] = sin(5*x[0]); }
  };

  // Sub domain for Dirichlet boundary condition
  class DirichletBoundary : public dolfin::SubDomain
  {
  public:
    bool inside(const dolfin::Array<double>& x, bool on_boundary) const
    { return x[0] < DOLFIN_EPS or x[0] > (1.0 - DOLFIN_EPS); }
  };

  std::tuple<std::shared_ptr<dolfin::PETScMatrix>,
    std::shared_ptr<dolfin::PETScVector>,
    std::shared_ptr<dolfin::Function>>
    problem(std::shared_ptr<const dolfin::Mesh> mesh)
  {
    dolfin::Timer t0("ZZZ FunctionSpace");
    auto V = std::make_shared<Poisson::FunctionSpace>(mesh);
    t0.stop();

    dolfin::Timer t1("ZZZ Assemble");

    // Define boundary condition
    auto u0 = std::make_shared<dolfin::Constant>(0.0);
    auto boundary = std::make_shared<DirichletBoundary>();
    auto bc = std::make_shared<dolfin::DirichletBC>(V, u0, boundary);

    // Define variational forms
    auto a = std::make_shared<Poisson::BilinearForm>(V, V);
    auto L = std::make_shared<Poisson::LinearForm>(V);

    // Attach coefficients
    auto f = std::make_shared<Source>();
    auto g = std::make_shared<dUdN>();
    L->f = f;
    L->g = g;

    // Create assembler
    dolfin::SystemAssembler assembler(a, L, {bc});

    // Assemble system
    auto A = std::make_shared<dolfin::PETScMatrix>();
    auto b = std::make_shared<dolfin::PETScVector>();
    assembler.assemble(*A, *b);

    t1.stop();

    // Create Function to hold solution
    auto u = std::make_shared<dolfin::Function>(V);

    return std::tuple<std::shared_ptr<dolfin::PETScMatrix>,
      std::shared_ptr<dolfin::PETScVector>,
      std::shared_ptr<dolfin::Function>>(A, b, u);
  }
}

// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
// Licensed under the MIT License. See LICENSE file in the project
// root for full license information.

#pragma once

#include <memory>
#include <utility>
#include <dolfin/common/Timer.h>
#include <dolfin/fem/DirichletBC.h>
#include <dolfin/fem/SystemAssembler.h>
#include <dolfin/function/Expression.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <dolfin/la/VectorSpaceBasis.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/SubDomain.h>

#include "Elasticity.h"

namespace elastic
{
  // Function to compute the near nullspace for elasticity - it is
  // made up of the six rigid body modes

  dolfin::la::VectorSpaceBasis build_near_nullspace(const dolfin::function::FunctionSpace& V,
                                                    const dolfin::la::PETScVector& x)
  {
    // Get subspaces
    auto V0 = V.sub({0});
    auto V1 = V.sub({1});
    auto V2 = V.sub({2});

    // Create vectors for nullspace basis
    std::vector<std::shared_ptr<dolfin::la::PETScVector>> basis(6);
    for (std::size_t i = 0; i < basis.size(); ++i)
      basis[i] = x.copy();

    // x0, x1, x2 translations
    V0->dofmap()->set(*basis[0], 1.0);
    V1->dofmap()->set(*basis[1], 1.0);
    V2->dofmap()->set(*basis[2], 1.0);

    // Rotations
    V0->set_x(*basis[3], -1.0, 1);
    V1->set_x(*basis[3],  1.0, 0);

    V0->set_x(*basis[4],  1.0, 2);
    V2->set_x(*basis[4], -1.0, 0);

    V2->set_x(*basis[5],  1.0, 1);
    V1->set_x(*basis[5], -1.0, 2);

    // Apply
    for (std::size_t i = 0; i < basis.size(); ++i)
      basis[i]->apply("add");

    // Create vector space and orthonormalize
    dolfin::la::VectorSpaceBasis vector_space(basis);
    vector_space.orthonormalize();
    return vector_space;
  }

  // Source term (right-hand side)
  class Source : public dolfin::function::Expression
  {
  public:

    Source() : Expression(3) {}

    void eval(dolfin::Array<double>& values,
              const dolfin::Array<double>& x) const
    {
      double dx = x[0] - 0.5;
      double dz = x[2] - 0.5;
      double r = dx*dx + dz*dz;

      values[0] = -dz*std::sqrt(r)*x[1];
      values[2] =  dx*std::sqrt(r)*x[1];

      values[1] = 1.0;
    }
  };

  // Bottom (x[1] = 0) surface
  class DirichletBoundary : public dolfin::mesh::SubDomain
  {
    bool inside(const Eigen::Array<double, Eigen::Dynamic>& x, bool on_boundary) const
    { return x[1] < 1.0e-8; }
  };

  std::tuple<std::shared_ptr<dolfin::la::PETScMatrix>,
    std::shared_ptr<dolfin::la::PETScVector>,
    std::shared_ptr<dolfin::function::Function>>
    problem(std::shared_ptr<const dolfin::mesh::Mesh> mesh)
  {
    dolfin::common::Timer t0("ZZZ FunctionSpace");
    auto V = std::make_shared<Elasticity::FunctionSpace>(mesh);
    t0.stop();

    dolfin::common::Timer t1("ZZZ Assemble");

    // Define boundary condition
    auto u0 = std::make_shared<dolfin::Constant>(0.0, 0.0, 0.0);
    auto boundary = std::make_shared<DirichletBoundary>();
    auto bc = std::make_shared<dolfin::fem::DirichletBC>(V, u0, boundary);

    // Define variational forms
    auto a = std::make_shared<Elasticity::BilinearForm>(V, V);
    auto L = std::make_shared<Elasticity::LinearForm>(V);

    // Elasticity parameters
    double E  = 1.0e6;
    double nu = 0.3;
    auto mu = std::make_shared<dolfin::Constant>(E/(2.0*(1.0 + nu)));
    auto lambda = std::make_shared<dolfin::Constant>(E*nu/((1.0 + nu)*(1.0 - 2.0*nu)));

    // Attach coefficients
    a->mu = mu; a->lmbda = lambda;
    auto f = std::make_shared<Source>();
    L->f = f;

    // Create assembler
    dolfin::SystemAssembler assembler(a, L, {bc});

    // Assemble system
    auto A = std::make_shared<dolfin::la::PETScMatrix>();
    auto b = std::make_shared<dolfin::la::PETScVector>();
    assembler.assemble(*A, *b);

    t1.stop();


    dolfin::Timer t2("ZZZ Create near-nullspace");
    // Create Function to hold solution
    auto u = std::make_shared<dolfin::function::Function>(V);

    // Build near-nullspace and attach to matrix
    dolfin::VectorSpaceBasis nullspace = build_near_nullspace(*V, *u->vector());
    A->set_near_nullspace(nullspace);
    t2.stop();

    return std::tuple<std::shared_ptr<dolfin::la::PETScMatrix>,
      std::shared_ptr<dolfin::la::PETScVector>,
      std::shared_ptr<dolfin::function::Function>>(A, b, u);
  }
}

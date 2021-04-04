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
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <utility>

std::tuple<dolfinx::la::PETScMatrix, dolfinx::la::PETScVector,
           std::shared_ptr<dolfinx::fem::Function<PetscScalar>>>
poisson::problem(std::shared_ptr<dolfinx::mesh::Mesh> mesh)
{
  dolfinx::common::Timer t0("ZZZ FunctionSpace");
  auto V = dolfinx::fem::create_functionspace(
      create_functionspace_form_Poisson_a, "u", mesh);
  t0.stop();

  // Define coefficients
  auto f = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  auto g = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);

  // Define variational forms
  auto L = dolfinx::fem::create_form<PetscScalar>(create_form_Poisson_L, {V},
                                                  {{"f", f}, {"g", g}}, {}, {});
  auto a = dolfinx::fem::create_form<PetscScalar>(create_form_Poisson_a, {V, V},
                                                  {}, {}, {});

  // Create matrices and vector, and assemble system
  dolfinx::la::PETScMatrix A(dolfinx::fem::create_matrix(*a), false);
  dolfinx::la::PETScVector b(*L->function_spaces()[0]->dofmap()->index_map,
                             L->function_spaces()[0]->dofmap()->index_map_bs());

  int rank = dolfinx::MPI::rank(mesh->mpi_comm());

  if (rank == 0)
    std::cout << "Log -  Create vectors \n";

  for (int i = 0; i < 10; i++)
  {
    VecSet(b.vec(), i);
    dolfinx::common::Timer t1("zzz PETSC Vector Scatter");
    VecGhostUpdateBegin(b.vec(), INSERT_VALUES, SCATTER_FORWARD);
    VecGhostUpdateEnd(b.vec(), INSERT_VALUES, SCATTER_FORWARD);
    t1.stop();
  }

  if (rank == 0)
    std::cout << "Log -  PETSC scatter \n";

  // Create Function to hold solution
  auto u = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  for (int i = 0; i < 10; i++)
  {
    dolfinx::common::Timer t2("zzz Dolfinx Vector Scatter");
    dolfinx::la::scatter_fwd(*u->x());
    t2.stop();
  }

  if (rank == 0)
    std::cout << "Log -  Dolfinx scatter \n";

  return {std::move(A), std::move(b), u};
}

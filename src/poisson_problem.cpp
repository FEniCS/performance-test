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
#include <iomanip>
#include <memory>
#include <utility>

std::tuple<dolfinx::la::PETScMatrix, dolfinx::la::PETScVector,
           std::shared_ptr<dolfinx::fem::Function<PetscScalar>>>
poisson::problem(std::shared_ptr<dolfinx::mesh::Mesh> mesh, bool use_petsc)
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
  auto u = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);

  int rank = dolfinx::MPI::rank(mesh->mpi_comm());
  int mpi_size = dolfinx::MPI::size(mesh->mpi_comm());

  if (rank == 0)
    std::cout << "Log -  Create vectors \n";

  std::cout << std::setprecision(8);

  std::vector<double> timers;
  if (use_petsc)
  {
    for (int i = 0; i < 10; i++)
    {
      VecSet(b.vec(), i);
      double t = MPI_Wtime();
      VecGhostUpdateBegin(b.vec(), INSERT_VALUES, SCATTER_FORWARD);
      VecGhostUpdateEnd(b.vec(), INSERT_VALUES, SCATTER_FORWARD);
      t = MPI_Wtime() - t;
      timers.push_back(t);
    }
    if (rank == 0)
      std::cout << "Log -  PETSC scatter \n";
  }
  else
  {
    for (int i = 0; i < 10; i++)
    {
      double t = MPI_Wtime();
      dolfinx::la::scatter_fwd(*u->x());
      t = MPI_Wtime() - t;
      timers.push_back(t);
    }
    if (rank == 0)
      std::cout << "Log -  Dolfinx scatter \n";
  }

  if (rank == 0)
  {

    std::string method = (use_petsc) ? "PETSc" : "Dolfinx";
    std::cout << "\n\n\n" <<mpi_size<< ", " << method << "\n\n";
    for (auto el : timers)
      std::cout << std::fixed << mpi_size << ", " << el << "\n";
    std::cout << "\n\n=====================\n";
  }

  return {std::move(A), std::move(b), u};
}
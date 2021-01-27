// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include "poisson_problem.h"
#include "Poisson.h"
#include <Eigen/Dense>
#include <cfloat>
#include <dolfinx/common/Timer.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <spmv.h>
#include <utility>

std::tuple<dolfinx::la::PETScMatrix, dolfinx::la::PETScVector,
           std::shared_ptr<dolfinx::fem::Function<PetscScalar>>>
poisson::problem(std::shared_ptr<dolfinx::mesh::Mesh> mesh)
{
  dolfinx::common::Timer t0("ZZZ FunctionSpace");

  auto V = dolfinx::fem::create_functionspace(
      create_functionspace_form_Poisson_a, "u", mesh);

  t0.stop();

  dolfinx::common::Timer t1("ZZZ Assemble");

  // Define boundary condition
  auto u0 = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  std::fill(u0->x()->mutable_array().begin(), u0->x()->mutable_array().end(),
            0.0);

  const std::vector<std::int32_t> bdofs
      = dolfinx::fem::locate_dofs_geometrical({*V}, [](auto& x) {
          return (x.row(0) < DBL_EPSILON or x.row(0) > 1.0 - DBL_EPSILON);
        });

  auto bc = std::make_shared<dolfinx::fem::DirichletBC<PetscScalar>>(u0, bdofs);

  // Define coefficients
  auto f = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  auto g = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
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

  // Define variational forms
  auto L = dolfinx::fem::create_form<PetscScalar>(create_form_Poisson_L, {V},
                                                  {{"f", f}, {"g", g}}, {}, {});
  auto a = dolfinx::fem::create_form<PetscScalar>(create_form_Poisson_a, {V, V},
                                                  {}, {}, {});

  // Create matrices and vector, and assemble system
  dolfinx::la::PETScMatrix A(dolfinx::fem::create_matrix(*a), false);
  dolfinx::la::PETScVector b(*L->function_spaces()[0]->dofmap()->index_map,
                             L->function_spaces()[0]->dofmap()->index_map_bs());

  MatZeroEntries(A.mat());
  dolfinx::common::Timer t2("ZZZ Assemble matrix");
  dolfinx::fem::assemble_matrix(dolfinx::la::PETScMatrix::add_fn(A.mat()), *a,
                                {bc});
  dolfinx::fem::add_diagonal(dolfinx::la::PETScMatrix::add_fn(A.mat()), *V,
                             {bc});
  MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);
  t2.stop();

  // Assembly into Eigen::SparseMatrix
  //-------------------------------------------------------

  auto im = V->dofmap()->index_map;
  int m = im->size_local() + im->num_ghosts();
  Eigen::SparseMatrix<PetscScalar> spmat(m, m);
  {
    std::vector<Eigen::Triplet<PetscScalar>> mat_data;
    std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                      const std::int32_t*, const PetscScalar*)>
        mat_add = [&mat_data](std::int32_t nr, const std::int32_t* rows,
                              const std::int32_t nc, const std::int32_t* cols,
                              const PetscScalar* data) {
          for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nc; ++j)
              mat_data.push_back(Eigen::Triplet<PetscScalar>(rows[i], cols[j],
                                                             data[i * nc + j]));

          return 0;
        };

    dolfinx::fem::assemble_matrix(mat_add, *a, {bc});
    assert(bc);
    dolfinx::fem::add_diagonal(mat_add, *V, {bc});
    spmat.setFromTriplets(mat_data.begin(), mat_data.end());
  }
  std::int64_t local_size = im->size_local();
  std::vector<std::int64_t> ghosts(im->ghosts().data(),
                                   im->ghosts().data() + im->num_ghosts());
  auto Aspmv = spmv::Matrix<PetscScalar>::create_matrix(
      mesh->mpi_comm(), spmat, local_size, local_size, ghosts, ghosts);

  //-------------------------------------------------------

  VecSet(b.vec(), 0.0);
  VecGhostUpdateBegin(b.vec(), INSERT_VALUES, SCATTER_FORWARD);
  VecGhostUpdateEnd(b.vec(), INSERT_VALUES, SCATTER_FORWARD);

  dolfinx::common::Timer t3("ZZZ Assemble vector");
  dolfinx::fem::assemble_vector_petsc(b.vec(), *L);
  dolfinx::fem::apply_lifting_petsc(b.vec(), {a}, {{bc}}, {}, 1.0);
  VecGhostUpdateBegin(b.vec(), ADD_VALUES, SCATTER_REVERSE);
  VecGhostUpdateEnd(b.vec(), ADD_VALUES, SCATTER_REVERSE);
  dolfinx::fem::set_bc_petsc(b.vec(), {bc}, nullptr);
  t3.stop();

  // Empty RHS
  Eigen::VectorXd bspmv(Aspmv.mat().rows());
  std::vector<int> rows(Aspmv.mat().rows());
  std::iota(rows.begin(), rows.end(), Aspmv.row_map()->global_offset());
  VecAssemblyBegin(b.vec());
  VecAssemblyEnd(b.vec());
  VecGetValues(b.vec(), Aspmv.mat().rows(), rows.data(), bspmv.data());

  double rtol = 1e-8;
  int max_its = 1000;
  auto [result, its] = spmv::cg(MPI_COMM_WORLD, Aspmv, bspmv, max_its, rtol);

  double rnorm = result.head(Aspmv.row_map()->local_size()).squaredNorm();
  double rnorm_sum;
  MPI_Allreduce(&rnorm, &rnorm_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  std::cout << "SPMV: Got result: " << std::sqrt(rnorm_sum) << " in " << its
            << " iterations\n";

  t1.stop();

  // Create Function to hold solution
  auto u = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);

  return {std::move(A), std::move(b), u};
}

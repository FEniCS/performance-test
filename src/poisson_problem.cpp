// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include "poisson_problem.h"
#include "Poisson.h"
#include <Eigen/Dense>
#include <MatrixMarket_Tpetra.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>

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
#include <dolfinx/la/SparsityPattern.h>
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

  dolfinx::common::Timer t1("ZZZ Assemble");

  // Define boundary condition
  auto u0 = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  u0->x()->array().setZero();

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

  dolfinx::la::SparsityPattern pattern
      = dolfinx::fem::create_sparsity_pattern(*a);
  pattern.assemble();
  const dolfinx::graph::AdjacencyList<std::int32_t>& diagonal_pattern
      = pattern.diagonal_pattern();
  const dolfinx::graph::AdjacencyList<std::int64_t>& off_diagonal_pattern
      = pattern.off_diagonal_pattern();

  std::vector<std::size_t> nnz(diagonal_pattern.num_nodes());
  for (int i = 0; i < diagonal_pattern.num_nodes(); ++i)
    nnz[i] = diagonal_pattern.num_links(i) + off_diagonal_pattern.num_links(i);

  auto comm = Tpetra::getDefaultComm();
  const std::vector<std::int64_t> global_indices0
      = V->dofmap()->index_map->global_indices();
  // Dumb copy (long long and int64_t should be the same, but compiler
  // complains)
  const std::vector<long long> global_indices(global_indices0.begin(),
                                              global_indices0.end());

  const Teuchos::ArrayView<const long long> global_index_view(
      global_indices.data(), global_indices.size());
  Teuchos::RCP<const Tpetra::Map<>> rowMap = Teuchos::rcp(new Tpetra::Map<>(
      V->dofmap()->index_map->size_global(), global_index_view, 0, comm));
  Teuchos::RCP<const Tpetra::Map<>> colMap = Teuchos::rcp(new Tpetra::Map<>(
      V->dofmap()->index_map->size_global(), global_index_view, 0, comm));

  const Teuchos::ArrayView<const long long> global_index_vec_view(
      global_indices.data(), V->dofmap()->index_map->size_local());
  Teuchos::RCP<const Tpetra::Map<>> vecMap = Teuchos::rcp(new Tpetra::Map<>(
      V->dofmap()->index_map->size_global(), global_index_vec_view, 0, comm));

  Teuchos::ArrayView<std::size_t> _nnz(nnz.data(), nnz.size());
  Teuchos::RCP<Tpetra::CrsGraph<>> crs_graph(
      new Tpetra::CrsGraph<>(vecMap, _nnz));

  for (std::size_t i = 0; i != diagonal_pattern.num_nodes(); ++i)
  {
    std::vector<long long> indices(diagonal_pattern.links(i).begin(),
                                   diagonal_pattern.links(i).end());
    indices.insert(indices.end(), off_diagonal_pattern.links(i).begin(),
                   off_diagonal_pattern.links(i).end());
    Teuchos::ArrayView<long long> _indices(indices.data(), indices.size());
    crs_graph->insertGlobalIndices(global_indices[i], _indices);
  }

  crs_graph->fillComplete(vecMap, vecMap);

  Tpetra::CrsMatrix<PetscScalar> A_Tpetra(crs_graph);

  std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                    const std::int32_t*, const PetscScalar*)>
      tpetra_insert
      = [&A_Tpetra, &global_indices](
            std::int32_t nr, const std::int32_t* rows, const std::int32_t nc,
            const std::int32_t* cols, const PetscScalar* data) {
          std::vector<long long> global_cols;
          for (std::int32_t i = 0; i < nc; ++i)
            global_cols.push_back(global_indices[cols[i]]);
          Teuchos::ArrayView<const long long> col_view(global_cols.data(), nc);
          for (std::int32_t i = 0; i < nr; ++i)
          {
            Teuchos::ArrayView<const double> data_view(data + i * nc, nc);
            A_Tpetra.sumIntoGlobalValues(global_indices[rows[i]], col_view,
                                         data_view);
          }
          return 0;
        };

  dolfinx::fem::assemble_matrix(tpetra_insert, *a, {bc});
  dolfinx::fem::add_diagonal(tpetra_insert, *V, {bc});

  A_Tpetra.fillComplete();

  Tpetra::MatrixMarket::Writer<Tpetra::CrsMatrix<>>::writeSparseFile(
      "testa.dat", A_Tpetra);

  double Tpetra_norm = A_Tpetra.getFrobeniusNorm();
  if (dolfinx::MPI::rank(mesh->mpi_comm()) == 0)
    std::cout << "NormA(Tpetra) = " << Tpetra_norm << "\n";

  Tpetra::Vector<PetscScalar> bdist_Tpetra(colMap), b_Tpetra(vecMap);
  Teuchos::ArrayRCP<PetscScalar> bdist_view = bdist_Tpetra.getDataNonConst();
  Teuchos::ArrayRCP<PetscScalar> b_view = b_Tpetra.getDataNonConst();
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 1> b_eigen(bdist_view.size());
  b_eigen.fill(0.0);
  dolfinx::fem::assemble_vector(Eigen::Ref<Eigen::VectorXd>(b_eigen), *L);
  dolfinx::fem::apply_lifting(Eigen::Ref<Eigen::VectorXd>(b_eigen), {a}, {{bc}},
                              {}, 1.0);
  dolfinx::fem::set_bc(Eigen::Ref<Eigen::VectorXd>(b_eigen), {bc});

  std::stringstream s;
  std::copy(b_eigen.data(), b_eigen.data() + b_eigen.size(), bdist_view.get());
  Tpetra::Export vec_export(colMap, vecMap);
  b_Tpetra.doExport(bdist_Tpetra, vec_export, Tpetra::CombineMode::ADD);
  s << "Norm[b](Tpetra) = " << b_Tpetra.norm2() << "\n";
  std::cout << s.str();

  // Create matrices and vector, and assemble system
  dolfinx::la::PETScMatrix A = dolfinx::fem::create_matrix(*a);
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

  double norm;
  MatNorm(A.mat(), NORM_FROBENIUS, &norm);
  if (dolfinx::MPI::rank(mesh->mpi_comm()) == 0)
    std::cout << "NormA(Petsc) = " << norm << "\n";

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

  VecNorm(b.vec(), NORM_2, &norm);
  std::cout << "Norm[b](Petsc) = " << norm << "\n";

  // Create Function to hold solution
  auto u = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);

  return {std::move(A), std::move(b), u};
}

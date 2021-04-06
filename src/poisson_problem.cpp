// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include "poisson_problem.h"
#include "Poisson.h"
#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>
#include <Eigen/Dense>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>

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
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <utility>

std::tuple<dolfinx::la::PETScMatrix, dolfinx::la::PETScVector,
           std::shared_ptr<dolfinx::fem::Function<PetscScalar>>>
poisson::problem(std::shared_ptr<dolfinx::mesh::Mesh> mesh)
{
  dolfinx::common::Timer t0("ZZZ FunctionSpace");
  std::stringstream s;

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
          constexpr double eps = 10.0 * std::numeric_limits<double>::epsilon();
          std::vector<bool> marked(x.shape[1]);
          std::transform(
              x.row(0).begin(), x.row(0).end(), marked.begin(),
              [](double x0) { return x0 < eps or std::abs(x0 - 1) < eps; });
          return marked;
        });

  auto bc = std::make_shared<dolfinx::fem::DirichletBC<PetscScalar>>(u0, bdofs);

  // Define coefficients
  auto f = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  auto g = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  f->interpolate([](auto& x) {
    std::vector<PetscScalar> f(x.shape[1]);
    std::transform(x.row(0).begin(), x.row(0).end(), x.row(1).begin(),
                   f.begin(), [](double x0, double x1) {
                     double dx
                         = (x0 - 0.5) * (x0 - 0.5) + (x1 - 0.5) * (x1 - 0.5);
                     return 10.0 * std::exp(-(dx) / 0.02);
                   });
    return f;
  });

  g->interpolate([](auto& x) {
    std::vector<PetscScalar> f(x.shape[1]);
    std::transform(x.row(0).begin(), x.row(0).end(), f.begin(),
                   [](double x0) { return std::sin(5 * x0); });
    return f;
  });

  // Define variational forms
  auto L = dolfinx::fem::create_form<PetscScalar>(create_form_Poisson_L, {V},
                                                  {{"f", f}, {"g", g}}, {}, {});
  auto a = dolfinx::fem::create_form<PetscScalar>(create_form_Poisson_a, {V, V},
                                                  {}, {}, {});

  Teuchos::RCP<const Teuchos::Comm<int>> comm
      = Teuchos::rcp(new Teuchos::MpiComm<int>(mesh->mpi_comm()));

  dolfinx::la::SparsityPattern pattern
      = dolfinx::fem::create_sparsity_pattern(*a);
  pattern.assemble();
  const dolfinx::graph::AdjacencyList<std::int32_t>& diagonal_pattern
      = pattern.diagonal_pattern();
  const dolfinx::graph::AdjacencyList<std::int32_t>& off_diagonal_pattern
      = pattern.off_diagonal_pattern();

  std::vector<std::size_t> nnz(diagonal_pattern.num_nodes());
  for (int i = 0; i < diagonal_pattern.num_nodes(); ++i)
    nnz[i] = diagonal_pattern.num_links(i) + off_diagonal_pattern.num_links(i);

  dolfinx::common::Timer tcre("Trilinos: create sparsity");
  std::vector<std::int64_t> global_indices
      = pattern.column_indices();

  const Teuchos::ArrayView<const std::int64_t> global_index_view(
      global_indices.data(), global_indices.size());
  Teuchos::RCP<const Tpetra::Map<std::int32_t, std::int64_t>> colMap
      = Teuchos::rcp(new Tpetra::Map<std::int32_t, std::int64_t>(
          V->dofmap()->index_map->size_global(), global_index_view, 0, comm));

  const Teuchos::ArrayView<const std::int64_t> global_index_vec_view(
      global_indices.data(), V->dofmap()->index_map->size_local());
  Teuchos::RCP<const Tpetra::Map<std::int32_t, std::int64_t>> vecMap
      = Teuchos::rcp(new Tpetra::Map<std::int32_t, std::int64_t>(
          V->dofmap()->index_map->size_global(), global_index_vec_view, 0,
          comm));

  Teuchos::ArrayView<std::size_t> _nnz(nnz.data(), nnz.size());
  Teuchos::RCP<Tpetra::CrsGraph<std::int32_t, std::int64_t>> crs_graph(
      new Tpetra::CrsGraph<std::int32_t, std::int64_t>(vecMap, colMap, _nnz));

  const std::int64_t nlocalrows = V->dofmap()->index_map->size_local();
  for (std::size_t i = 0; i != diagonal_pattern.num_nodes(); ++i)
  {
    std::vector<std::int32_t> indices(diagonal_pattern.links(i).begin(),
                                      diagonal_pattern.links(i).end());
    for (std::int32_t q : off_diagonal_pattern.links(i))
      indices.push_back(q);

    Teuchos::ArrayView<std::int32_t> _indices(indices.data(), indices.size());
    crs_graph->insertLocalIndices(i, _indices);
  }

  crs_graph->fillComplete();
  tcre.stop();

  Teuchos::RCP<Tpetra::CrsMatrix<PetscScalar, std::int32_t, std::int64_t>>
      A_Tpetra = Teuchos::rcp(
          new Tpetra::CrsMatrix<PetscScalar, std::int32_t, std::int64_t>(
              crs_graph));

  // Temp storage for off-process row indices
  std::vector<std::int64_t> global_cols;

  std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                    const std::int32_t*, const PetscScalar*)>
      tpetra_insert
      = [&A_Tpetra, &global_indices, &global_cols, &nlocalrows](
            std::int32_t nr, const std::int32_t* rows, const std::int32_t nc,
            const std::int32_t* cols, const PetscScalar* data) {
          for (std::int32_t i = 0; i < nr; ++i)
          {
            Teuchos::ArrayView<const double> data_view(data + i * nc, nc);
            if (rows[i] < nlocalrows)
            {
              Teuchos::ArrayView<const int> col_view(cols, nc);
              int nvalid
                  = A_Tpetra->sumIntoLocalValues(rows[i], col_view, data_view);
              if (nvalid != nc)
                throw std::runtime_error(
                    "Inserted " + std::to_string(nvalid) + "/"
                    + std::to_string(nc)
                    + " on row:" + std::to_string(global_indices[rows[i]]));
            }
            else
            {
              global_cols.resize(nc);
              for (int j = 0; j < nc; ++j)
                global_cols[j] = global_indices[cols[j]];
              int nvalid = A_Tpetra->sumIntoGlobalValues(
                  global_indices[rows[i]], global_cols, data_view);
              if (nvalid != nc)
                throw std::runtime_error(
                    "Inserted " + std::to_string(nvalid) + "/"
                    + std::to_string(nc)
                    + " on row:" + std::to_string(global_indices[rows[i]]));
            }
          }
          return 0;
        };

  dolfinx::common::Timer tassm("Trilinos: assemble matrix");
  dolfinx::fem::assemble_matrix(tpetra_insert, *a, {bc});
  dolfinx::fem::add_diagonal(tpetra_insert, *V, {bc});

  A_Tpetra->fillComplete(vecMap, vecMap);
  tassm.stop();

  double Tpetra_norm = A_Tpetra->getFrobeniusNorm();
  if (dolfinx::MPI::rank(mesh->mpi_comm()) == 0)
    s << "NormA(Tpetra) = " << Tpetra_norm << "\n";

  using MV = Tpetra::MultiVector<PetscScalar, std::int32_t, std::int64_t>;
  using OP = Tpetra::Operator<PetscScalar, std::int32_t, std::int64_t>;

  dolfinx::common::Timer tassv("Trilinos: assemble vector");
  Teuchos::RCP<MV> b_Tpetra(new MV(vecMap, 1));
  Teuchos::RCP<MV> x_Tpetra(new MV(vecMap, 1));

  {
    // Assemble RHS and gather ghost entries
    Teuchos::RCP<MV> bdist_Tpetra(new MV(colMap, 1));
    Teuchos::ArrayRCP<PetscScalar> bdist_view
        = bdist_Tpetra->getDataNonConst(0);
    tcb::span<PetscScalar> b_(bdist_view.get(), bdist_view.size());
    for (PetscScalar& v : bdist_view)
      v = 0.0;

    dolfinx::fem::assemble_vector(b_, *L);
    dolfinx::fem::apply_lifting(b_, {a}, {{bc}}, {}, 1.0);

    Tpetra::Export<std::int32_t, std::int64_t> vec_export(colMap, vecMap);
    b_Tpetra->doExport(*bdist_Tpetra, vec_export, Tpetra::CombineMode::ADD);

    Teuchos::ArrayRCP<PetscScalar> bdist_viewt = b_Tpetra->getDataNonConst(0);
    tcb::span<PetscScalar> bt_(bdist_viewt.get(), bdist_viewt.size());
    dolfinx::fem::set_bc(bt_, {bc});
  }
  tassv.stop();

  double norm2;
  Teuchos::ArrayView<double> norm_view(&norm2, 1);
  b_Tpetra->norm2(norm_view);
  if (dolfinx::MPI::rank(mesh->mpi_comm()) == 0)
    s << "Norm[b](Tpetra) = " << norm2 << "\n";

  dolfinx::common::Timer ttri("Trilinos: solve");

  // Muelu preconditioner, to be constructed from a Tpetra Operator
  // or Matrix
  Teuchos::RCP<Teuchos::ParameterList> muelu_paramList(
      new Teuchos::ParameterList);
  muelu_paramList->set("problem: type", "Poisson-3D");
  Teuchos::RCP<MueLu::TpetraOperator<PetscScalar, std::int32_t, std::int64_t>>
      muelu_prec = MueLu::CreateTpetraPreconditioner(
          Teuchos::rcp_dynamic_cast<
              Tpetra::Operator<PetscScalar, std::int32_t, std::int64_t>>(
              A_Tpetra),
          *muelu_paramList);

  Teuchos::RCP<Teuchos::ParameterList> solver_paramList(
      new Teuchos::ParameterList);
  solver_paramList->set("Convergence Tolerance", 1e-8);
  solver_paramList->set("Verbosity", Belos::Warnings | Belos::IterationDetails
                                         | Belos::StatusTestDetails
                                         | Belos::TimingDetails
                                         | Belos::FinalSummary);
  solver_paramList->set("Output Style", (int)Belos::Brief);
  solver_paramList->set("Output Frequency", 1);
  Belos::SolverFactory<PetscScalar, MV, OP> factory;
  Teuchos::RCP<Belos::SolverManager<PetscScalar, MV, OP>> belos_solver
      = factory.create("CG", solver_paramList);

  Teuchos::RCP<Belos::LinearProblem<double, MV, OP>> problem(
      new Belos::LinearProblem<double, MV, OP>);
  problem->setOperator(A_Tpetra);
  problem->setLeftPrec(muelu_prec);
  problem->setProblem(x_Tpetra, b_Tpetra);
  belos_solver->setProblem(problem);
  belos_solver->solve();
  x_Tpetra->norm2(norm_view);
  if (dolfinx::MPI::rank(mesh->mpi_comm()) == 0)
    s << "Norm[x](Tpetra) = " << norm2 << "\n";
  ttri.stop();

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

  double norm;
  MatNorm(A.mat(), NORM_FROBENIUS, &norm);
  if (dolfinx::MPI::rank(mesh->mpi_comm()) == 0)
    s << "NormA(Petsc) = " << norm << "\n";

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
  s << "Norm[b](Petsc) = " << norm << "\n";

  // Create Function to hold solution
  auto u = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  std::cout << s.str();
  return {std::move(A), std::move(b), u};
}

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
      = pattern.index_map(1)->global_indices();

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

  std::vector<std::vector<std::pair<std::int64_t, PetscScalar>>> remote_rows(
      V->dofmap()->index_map->num_ghosts());

  Teuchos::RCP<Tpetra::CrsMatrix<PetscScalar, std::int32_t, std::int64_t>>
      A_Tpetra = Teuchos::rcp(
          new Tpetra::CrsMatrix<PetscScalar, std::int32_t, std::int64_t>(
              crs_graph));
  std::vector<std::int64_t> global_cols;
  std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                    const std::int32_t*, const PetscScalar*)>
      tpetra_insert
      = [&A_Tpetra, &global_indices, &global_cols, &nlocalrows, &remote_rows](
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
              for (std::int32_t j = 0; j < nc; ++j)
              {
                global_cols[j] = global_indices[cols[j]];
                remote_rows[rows[i] - nlocalrows].push_back(
                    {global_cols[j], data[i * nc + j]});
              }
            }
          }
          return 0;
        };

  dolfinx::common::Timer tassm("Trilinos: assemble matrix");
  dolfinx::fem::assemble_matrix(tpetra_insert, *a, {bc});
  dolfinx::fem::add_diagonal(tpetra_insert, *V, {bc});

  // SEND CACHED ROWS TO OWNING PROCESS

  dolfinx::common::Timer tnbr("Assembly: Send off-process rows");
  dolfinx::common::Timer tnbr2("Assembly: Create send data");

  // Condense cache (sort and sum values on each row)
  int mpi_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  const std::vector<int> ghost_owners
      = V->dofmap()->index_map->ghost_owner_rank();
  const std::vector<std::int64_t>& ghost_row = V->dofmap()->index_map->ghosts();

  MPI_Comm neighbor_comm = V->dofmap()->index_map->comm(
      dolfinx::common::IndexMap::Direction::reverse);
  auto [src_nbr, dest_nbr] = dolfinx::MPI::neighbors(neighbor_comm);
  int nnbr = dest_nbr.size();
  std::map<int, int> nbrmap;
  for (std::size_t i = 0; i < dest_nbr.size(); ++i)
    nbrmap.insert({dest_nbr[i], i});

  std::vector<std::vector<std::int64_t>> send_data_int(nnbr);
  std::vector<std::vector<PetscScalar>> send_data_scalar(nnbr);
  std::map<std::int64_t, PetscScalar> row_sum;
  for (std::size_t i = 0; i < ghost_row.size(); ++i)
  {
    std::vector<std::pair<std::int64_t, PetscScalar>>& row = remote_rows[i];
    row_sum.clear();
    for (const std::pair<std::int64_t, PetscScalar>& q : row)
      row_sum[q.first] += q.second;
    std::vector<std::int64_t>& sd = send_data_int[nbrmap[ghost_owners[i]]];
    std::vector<PetscScalar>& ss = send_data_scalar[nbrmap[ghost_owners[i]]];

    sd.push_back(ghost_row[i]);
    sd.push_back(row_sum.size());
    for (const std::pair<std::int64_t, PetscScalar>& q : row_sum)
    {
      sd.push_back(q.first);
      ss.push_back(q.second);
    }
  }

  auto send_int = dolfinx::graph::AdjacencyList<std::int64_t>(send_data_int);
  auto send_scalar
      = dolfinx::graph::AdjacencyList<PetscScalar>(send_data_scalar);

  tnbr2.stop();
  dolfinx::common::Timer tnbr3("Assembly: alltoall");

  auto recv_data_int = dolfinx::MPI::neighbor_all_to_all(
      neighbor_comm, send_int.offsets(), send_int.array());
  auto recv_data_scalar = dolfinx::MPI::neighbor_all_to_all(
      neighbor_comm, send_scalar.offsets(), send_scalar.array());
  MPI_Barrier(MPI_COMM_WORLD);
  tnbr3.stop();

  dolfinx::common::Timer tnbr4("Assembly: collect data");

  std::shared_ptr<const dolfinx::common::IndexMap> p1 = pattern.index_map(1);
  const std::int64_t col_max = p1->local_range()[1];
  const std::int64_t col_min = p1->local_range()[0];
  std::map<std::int64_t, std::int32_t> column_global_to_local;
  for (std::size_t i = 0; i < p1->ghosts().size(); ++i)
    column_global_to_local.insert({p1->ghosts()[i], i + (col_max - col_min)});

  for (int p = 0; p < recv_data_int.num_nodes(); ++p)
  {
    auto pdata = recv_data_int.links(p);
    std::vector<std::int64_t> global_rows;
    std::vector<int> num_cols;
    for (int i = 0; i < recv_data_int.num_links(p); i += (pdata[i + 1] + 2))
    {
      global_rows.push_back(pdata[i]);
      num_cols.push_back(pdata[i + 1]);
    }
    std::vector<std::int32_t> local_rows
        = V->dofmap()->index_map->global_to_local(global_rows);

    int c = 2;
    int d = 0;
    for (std::size_t i = 0; i < local_rows.size(); ++i)
    {
      const int num_col = num_cols[i];

      Teuchos::ArrayView<const PetscScalar> data_view(
          recv_data_scalar.links(p).data() + d, num_col);
      std::vector<std::int32_t> col_local(num_col);
      for (int j = 0; j < num_col; ++j)
      {
        const std::int64_t gi = recv_data_int.links(p)[c + j];
        if (gi >= col_min and gi < col_max)
          col_local[j] = gi - col_min;
        else
          col_local[j] = column_global_to_local[gi];
      }
      int nvalid
          = A_Tpetra->sumIntoLocalValues(local_rows[i], col_local, data_view);
      c += (num_col + 2);
      d += num_col;
    }
  }
  tnbr4.stop();
  tnbr.stop();

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

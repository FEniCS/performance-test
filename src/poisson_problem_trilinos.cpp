// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include "poisson_problem_trilinos.h"
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
#include <dolfinx/fem/utils.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <utility>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

std::tuple<dolfinx::la::Vector<PetscScalar>,
           std::shared_ptr<dolfinx::fem::Function<PetscScalar>>,
           std::function<int(dolfinx::fem::Function<PetscScalar>&,
                             const dolfinx::la::Vector<PetscScalar>&)>>
poisson_trilinos::problem(std::shared_ptr<dolfinx::mesh::Mesh> mesh)
{
  dolfinx::common::Timer t0("ZZZ FunctionSpace");
  std::stringstream s;

  auto V = dolfinx::fem::create_functionspace(
      create_functionspace_form_Poisson_a, "u", mesh);

  t0.stop();

  dolfinx::common::Timer t1("ZZZ Assemble");

  dolfinx::common::Timer t2("ZZZ Create boundary conditions");
  // Define boundary condition
  auto u0 = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  std::fill(u0->x()->mutable_array().begin(), u0->x()->mutable_array().end(),
            0.0);

  const std::vector<std::int32_t> bdofs = dolfinx::fem::locate_dofs_geometrical(
      {*V}, [](const xt::xtensor<double, 2>& x) -> xt::xtensor<bool, 1> {
        auto x0 = xt::row(x, 0);
        return xt::isclose(x0, 0.0) or xt::isclose(x0, 1.0);
      });

  auto bc = std::make_shared<dolfinx::fem::DirichletBC<PetscScalar>>(u0, bdofs);
  t2.stop();

  // Define coefficients
  dolfinx::common::Timer t3("ZZZ Create RHS function");
  auto f = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  auto g = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  f->interpolate(
      [](const xt::xtensor<double, 2>& x) -> xt::xarray<PetscScalar> {
        auto dx
            = xt::square(xt::row(x, 0) - 0.5) + xt::square(xt::row(x, 1) - 0.5);
        return 10 * xt::exp(-(dx) / 0.02);
      });

  g->interpolate(
      [](const xt::xtensor<double, 2>& x) -> xt::xarray<PetscScalar> {
        return xt::sin(5.0 * xt::row(x, 0));
      });
  t3.stop();

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
  std::vector<std::int64_t> global_indices = pattern.column_indices();

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
  dolfinx::la::Vector<PetscScalar> bx(
      L->function_spaces()[0]->dofmap()->index_map,
      L->function_spaces()[0]->dofmap()->index_map_bs());
  tcb::span<PetscScalar> b_(bx.mutable_array().data(),
                            bx.mutable_array().size());

  std::fill(b_.begin(), b_.end(), 0.0);
  dolfinx::fem::assemble_vector(b_, *L);
  dolfinx::fem::apply_lifting(b_, {a}, {{bc}}, {}, 1.0);
  dolfinx::la::scatter_rev(bx, dolfinx::common::IndexMap::Mode::add);
  dolfinx::fem::set_bc(b_, {bc});

  const int size_local = V->dofmap()->index_map->size_local();
  double local_norm = std::transform_reduce(
      b_.data(), b_.data() + size_local, 0.0, std::plus<double>(),
      [](PetscScalar val) { return std::norm(val); });

  double global_norm;
  MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM,
                bx.map()->comm());

  if (dolfinx::MPI::rank(mesh->mpi_comm()) == 0)
    s << "Norm[b](Tpetra) = " << std::sqrt(global_norm) << "\n";

  //---

  std::cout << s.str();

  // Create Function to hold solution
  auto u = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);

  std::function<int(dolfinx::fem::Function<PetscScalar>&,
                    const dolfinx::la::Vector<PetscScalar>&)>
      solver_function = [A_Tpetra,
                         vecMap](dolfinx::fem::Function<PetscScalar>& u,
                                 const dolfinx::la::Vector<PetscScalar>& b) {
        dolfinx::common::Timer ttri("Trilinos: solve");

        // FIXME: how to wrap memory with MultiVector? - this is a copy
        const Teuchos::ArrayView<const PetscScalar> b_view(b.array().data(),
                                                           b.array().size());
        Teuchos::RCP<MV> b_Tpetra(new MV(vecMap, b_view, b.array().size(), 1));

        // Muelu preconditioner, to be constructed from a Tpetra Operator
        // or Matrix
        Teuchos::RCP<Teuchos::ParameterList> muelu_paramList(
            new Teuchos::ParameterList);
        muelu_paramList->set("problem: type", "Poisson-3D");
        Teuchos::RCP<
            MueLu::TpetraOperator<PetscScalar, std::int32_t, std::int64_t>>
            muelu_prec = MueLu::CreateTpetraPreconditioner(
                Teuchos::rcp_dynamic_cast<
                    Tpetra::Operator<PetscScalar, std::int32_t, std::int64_t>>(
                    A_Tpetra),
                *muelu_paramList);

        Teuchos::RCP<Teuchos::ParameterList> solver_paramList(
            new Teuchos::ParameterList);
        solver_paramList->set("Convergence Tolerance", 1e-8);
        solver_paramList->set("Verbosity",
                              Belos::Warnings | Belos::IterationDetails
                                  | Belos::StatusTestDetails
                                  | Belos::TimingDetails | Belos::FinalSummary);
        solver_paramList->set("Output Style", (int)Belos::Brief);
        solver_paramList->set("Output Frequency", 1);
        Belos::SolverFactory<PetscScalar, MV, OP> factory;
        Teuchos::RCP<Belos::SolverManager<PetscScalar, MV, OP>> belos_solver
            = factory.create("CG", solver_paramList);

        Teuchos::RCP<Belos::LinearProblem<double, MV, OP>> problem(
            new Belos::LinearProblem<double, MV, OP>);
        problem->setOperator(A_Tpetra);
        problem->setLeftPrec(muelu_prec);

        Teuchos::RCP<MV> x_Tpetra(new MV(vecMap, 1));
        problem->setProblem(x_Tpetra, b_Tpetra);
        belos_solver->setProblem(problem);
        belos_solver->solve();

        // Copy out solution vector
        std::copy(x_Tpetra->getData(0).begin(), x_Tpetra->getData(0).end(),
                  u.x()->mutable_array().data());

        const int num_iters = belos_solver->getNumIters();
        std::cout << "num iters = " << num_iters << "\n";

        return num_iters;
      };

  return {std::move(bx), u, solver_function};
}

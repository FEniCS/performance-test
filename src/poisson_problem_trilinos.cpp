// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include "poisson_problem_trilinos.h"
#include "Poisson.h"
#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>
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

using T = PetscScalar;

std::tuple<std::shared_ptr<dolfinx::la::Vector<T>>,
           std::shared_ptr<dolfinx::fem::Function<T>>,
           std::function<int(dolfinx::fem::Function<T>&,
                             const dolfinx::la::Vector<T>&)>>
poisson_trilinos::problem(std::shared_ptr<dolfinx::mesh::Mesh<double>> mesh,
                          int order)
{
  dolfinx::common::Timer t0("ZZZ FunctionSpace");
  std::stringstream s;

  std::vector fs_poisson_a
      = {functionspace_form_Poisson_a1, functionspace_form_Poisson_a2,
         functionspace_form_Poisson_a3};

  auto V = std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace(*fs_poisson_a.at(order - 1), "v_0", mesh));

  t0.stop();

  dolfinx::common::Timer t1("ZZZ Assemble");

  dolfinx::common::Timer t2("ZZZ Create boundary conditions");
  // Define boundary condition
  auto u0 = std::make_shared<dolfinx::fem::Function<T>>(V);
  std::fill(u0->x()->mutable_array().begin(), u0->x()->mutable_array().end(),
            0.0);

  // Find facets with bc applied
  const int tdim = mesh->topology()->dim();
  const std::vector<std::int32_t> bc_facets = mesh::locate_entities(
      *mesh, tdim - 1,
      [](auto x)
      {
        constexpr double eps = 1.0e-8;
        std::vector<std::int8_t> marker(x.extent(1), false);
        for (std::size_t p = 0; p < x.extent(1); ++p)
        {
          double x0 = x(0, p);
          if (std::abs(x0) < eps or std::abs(x0 - 1) < eps)
            marker[p] = true;
        }
        return marker;
      });

  // Find constrained dofs
  const std::vector<std::int32_t> bdofs = fem::locate_dofs_topological(
      *V->mesh()->topology_mutable(), *V->dofmap(), tdim - 1, bc_facets);

  auto bc = std::make_shared<fem::DirichletBC<T>>(u0, bdofs);
  t2.stop();

  // Define coefficients
  dolfinx::common::Timer t3("ZZZ Create RHS function");
  auto f = std::make_shared<dolfinx::fem::Function<T>>(V);
  auto g = std::make_shared<dolfinx::fem::Function<T>>(V);
  f->interpolate(
      [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
      {
        std::vector<T> v(x.extent(1));
        for (std::size_t p = 0; p < x.extent(1); ++p)
        {
          double dx = x(0, p) - 0.5;
          double dy = x(1, p) - 0.5;
          double dr = dx * dx + dy * dy;
          v[p] = 10 * std::exp(-dr / 0.02);
        }

        return {std::move(v), {v.size()}};
      });
  g->interpolate(
      [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
      {
        std::vector<T> f(x.extent(1));
        for (std::size_t p = 0; p < x.extent(1); ++p)
          f[p] = std::sin(5 * x(0, p));
        return {f, {f.size()}};
      });
  t3.stop();

  std::vector form_poisson_L
      = {form_Poisson_L1, form_Poisson_L2, form_Poisson_L3};
  std::vector form_poisson_a
      = {form_Poisson_a1, form_Poisson_a2, form_Poisson_a3};

  // Define variational forms
  auto L = std::make_shared<fem::Form<T>>(fem::create_form<T>(
      *form_poisson_L.at(order - 1), {V}, {{"w0", f}, {"w1", g}}, {}, {}));
  auto a = std::make_shared<fem::Form<T>>(fem::create_form<T>(
      *form_poisson_a.at(order - 1), {V, V},
      std::vector<std::shared_ptr<const fem::Function<T>>>{}, {}, {}));

  Teuchos::RCP<const Teuchos::Comm<int>> comm
      = Teuchos::rcp(new Teuchos::MpiComm<int>(mesh->comm()));

  dolfinx::la::SparsityPattern pattern
      = dolfinx::fem::create_sparsity_pattern(*a);
  pattern.finalize();
  auto [sp_edges, sp_offsets] = pattern.graph();
  const std::size_t nlocal = pattern.index_map(0)->size_local();

  // Get nnz on each local row
  std::vector<std::size_t> nnz(nlocal);
  for (int i = 0; i < nlocal; ++i)
    nnz[i] = sp_offsets[i + 1] - sp_offsets[i];

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

  for (std::size_t i = 0; i != nlocal; ++i)
  {
    Teuchos::ArrayView<const std::int32_t> _indices(
        sp_edges.data() + sp_offsets[i], nnz[i]);
    crs_graph->insertLocalIndices(i, _indices);
  }

  crs_graph->fillComplete();
  tcre.stop();

  Teuchos::RCP<Tpetra::CrsMatrix<T, std::int32_t, std::int64_t>> A_Tpetra
      = Teuchos::rcp(
          new Tpetra::CrsMatrix<T, std::int32_t, std::int64_t>(crs_graph));

  // Temp storage for off-process row indices
  std::vector<std::int64_t> global_cols;

  std::function<int(const std::span<const std::int32_t>&,
                    const std::span<const std::int32_t>&,
                    const std::span<const T>&)>
      tpetra_insert = [&A_Tpetra, &global_indices, &global_cols,
                       &nlocal](const std::span<const std::int32_t>& rows,
                                const std::span<const std::int32_t>& cols,
                                const std::span<const T>& data)
  {
    const std::size_t nr = rows.size();
    const std::size_t nc = cols.size();
    for (std::int32_t i = 0; i < nr; ++i)
    {
      Teuchos::ArrayView<const double> data_view(data.data() + i * nc, nc);
      if (rows[i] < nlocal)
      {
        Teuchos::ArrayView<const int> col_view(cols.data(), nc);
        int nvalid = A_Tpetra->sumIntoLocalValues(rows[i], col_view, data_view);
        if (nvalid != nc)
          throw std::runtime_error("Inserted " + std::to_string(nvalid) + "/"
                                   + std::to_string(nc) + " on row:"
                                   + std::to_string(global_indices[rows[i]]));
      }
      else
      {
        global_cols.resize(nc);
        for (int j = 0; j < nc; ++j)
          global_cols[j] = global_indices[cols[j]];
        int nvalid = A_Tpetra->sumIntoGlobalValues(global_indices[rows[i]],
                                                   global_cols, data_view);
        if (nvalid != nc)
          throw std::runtime_error("Inserted " + std::to_string(nvalid) + "/"
                                   + std::to_string(nc) + " on row:"
                                   + std::to_string(global_indices[rows[i]]));
      }
    }
    return 0;
  };

  auto tpetra_set = [&A_Tpetra, &global_indices,
                     &nlocal](const std::span<const std::int32_t>& rows,
                              const std::span<const std::int32_t>& cols,
                              const std::span<const T>& data)
  {
    const std::size_t nr = rows.size();
    const std::size_t nc = cols.size();
    if (rows[0] >= nlocal or nr > 1 or nc > 1)
      throw std::runtime_error("Error setting diagonal");
    Teuchos::ArrayView<const int> col_view(cols.data(), 1);
    Teuchos::ArrayView<const double> data_view(data.data(), 1);
    int nvalid = A_Tpetra->replaceLocalValues(rows[0], col_view, data_view);
    if (nvalid != nc)
      throw std::runtime_error("Inserted " + std::to_string(nvalid) + "/"
                               + std::to_string(nc) + " on row:"
                               + std::to_string(global_indices[rows[0]]));

    return 0;
  };

  dolfinx::common::Timer tassm("Trilinos: assemble matrix");
  const std::vector constants_a = fem::pack_constants(*a);
  auto coeffs_a = fem::allocate_coefficient_storage(*a);
  fem::pack_coefficients(*a, coeffs_a);
  dolfinx::fem::assemble_matrix<T>(tpetra_insert, *a, constants_a,
                                   fem::make_coefficients_span(coeffs_a), {bc});
  dolfinx::fem::set_diagonal<T>(tpetra_set, *V, {bc});

  A_Tpetra->fillComplete(vecMap, vecMap);
  tassm.stop();

  double Tpetra_norm = A_Tpetra->getFrobeniusNorm();
  if (dolfinx::MPI::rank(mesh->comm()) == 0)
    s << "NormA(Tpetra) = " << Tpetra_norm << "\n";

  using MV = Tpetra::MultiVector<T, std::int32_t, std::int64_t>;
  using OP = Tpetra::Operator<T, std::int32_t, std::int64_t>;

  // Create la::Vector
  la::Vector<T> b(L->function_spaces()[0]->dofmap()->index_map,
                  L->function_spaces()[0]->dofmap()->index_map_bs());
  b.set(0);

  common::Timer t5("ZZZ Assemble vector");
  const std::vector constants_L = fem::pack_constants(*L);
  auto coeffs_L = fem::allocate_coefficient_storage(*L);
  fem::pack_coefficients(*L, coeffs_L);
  fem::assemble_vector<T>(b.mutable_array(), *L, constants_L,
                          fem::make_coefficients_span(coeffs_L));
  fem::apply_lifting<T, double>(b.mutable_array(), {a}, {constants_L},
                                {fem::make_coefficients_span(coeffs_L)}, {{bc}},
                                {}, 1.0);
  b.scatter_rev(std::plus<>());
  fem::set_bc<T, double>(b.mutable_array(), {bc});
  t5.stop();

  const int size_local = V->dofmap()->index_map->size_local();
  double local_norm = std::transform_reduce(
      b.array().data(), b.array().data() + size_local, 0.0, std::plus<double>(),
      [](T val) { return std::norm(val); });

  double global_norm;
  MPI_Allreduce(&local_norm, &global_norm, 1, MPI_DOUBLE, MPI_SUM,
                b.index_map()->comm());

  if (dolfinx::MPI::rank(mesh->comm()) == 0)
    s << "Norm[b](Tpetra) = " << std::sqrt(global_norm) << "\n";

  //---

  std::cout << s.str();

  // Create Function to hold solution
  auto u = std::make_shared<dolfinx::fem::Function<T>>(V);

  std::function<int(dolfinx::fem::Function<T>&, const dolfinx::la::Vector<T>&)>
      solver_function = [A_Tpetra, vecMap](dolfinx::fem::Function<T>& u,
                                           const dolfinx::la::Vector<T>& b)
  {
    dolfinx::common::Timer ttri("Trilinos: solve");

    // FIXME: how to wrap memory with MultiVector? - this is a copy
    const Teuchos::ArrayView<const T> b_view(b.array().data(),
                                             b.array().size());
    Teuchos::RCP<MV> b_Tpetra(new MV(vecMap, b_view, b.array().size(), 1));

    // Muelu preconditioner, to be constructed from a Tpetra Operator
    // or Matrix
    Teuchos::RCP<Teuchos::ParameterList> muelu_paramList(
        new Teuchos::ParameterList);
    muelu_paramList->set("problem: type", "Poisson-3D");
    Teuchos::RCP<MueLu::TpetraOperator<T, std::int32_t, std::int64_t>>
        muelu_prec = MueLu::CreateTpetraPreconditioner(
            Teuchos::rcp_dynamic_cast<
                Tpetra::Operator<T, std::int32_t, std::int64_t>>(A_Tpetra),
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
    Belos::SolverFactory<T, MV, OP> factory;
    Teuchos::RCP<Belos::SolverManager<T, MV, OP>> belos_solver
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

  return {std::make_shared<la::Vector<T>>(std::move(b)), u, solver_function};
}

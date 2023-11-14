// Copyright (C) 2021 Chris N. Richardson
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include "elasticity_problem_trilinos.h"
#include "Elasticity.h"
#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <dolfinx/common/Timer.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <utility>

#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>

using T = PetscScalar;

std::tuple<std::shared_ptr<dolfinx::la::Vector<T>>,
           std::shared_ptr<dolfinx::fem::Function<T>>,
           std::function<int(dolfinx::fem::Function<T>&,
                             const dolfinx::la::Vector<T>&)>>
elasticity_trilinos::problem(std::shared_ptr<dolfinx::mesh::Mesh<double>> mesh,
                             int order)
{
  dolfinx::common::Timer t0("ZZZ FunctionSpace");

  std::vector fs_elasticity
      = {functionspace_form_Elasticity_a1, functionspace_form_Elasticity_a2,
         functionspace_form_Elasticity_a3};
  auto V = std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace(*fs_elasticity.at(order - 1), "v_0", mesh));

  t0.stop();

  dolfinx::common::Timer t0a("ZZZ Create boundary conditions");

  // Define boundary condition
  auto u0 = std::make_shared<dolfinx::fem::Function<T>>(V);
  u0->x()->set(0);

  const int tdim = mesh->topology()->dim();

  // Find facets with bc applied
  const std::vector<std::int32_t> bc_facets = mesh::locate_entities(
      *mesh, tdim - 1,
      [](auto x)
      {
        constexpr double eps = 1.0e-8;
        std::vector<std::int8_t> marker(x.extent(1), false);
        for (std::size_t p = 0; p < x.extent(1); ++p)
        {
          double x1 = x(1, p);
          if (std::abs(x1) < eps)
            marker[p] = true;
        }
        return marker;
      });

  // Find constrained dofs
  const std::vector<std::int32_t> bdofs = fem::locate_dofs_topological(
      *V->mesh()->topology_mutable(), *V->dofmap(), tdim - 1, bc_facets);

  // Bottom (x[1] = 0) surface
  auto bc = std::make_shared<fem::DirichletBC<T>>(u0, bdofs);
  t0a.stop();

  dolfinx::common::Timer t0b("ZZZ Create RHS function");

  // Define coefficients
  auto f = std::make_shared<fem::Function<T>>(V);
  f->interpolate(
      [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
      {
        std::vector<T> vdata(x.extent(0) * x.extent(1));
        namespace stdex
            = MDSPAN_IMPL_STANDARD_NAMESPACE::MDSPAN_IMPL_PROPOSED_NAMESPACE;
        MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
            T,
            MDSPAN_IMPL_STANDARD_NAMESPACE::extents<
                std::size_t, 3, MDSPAN_IMPL_STANDARD_NAMESPACE::dynamic_extent>>
            v(vdata.data(), x.extent(0), x.extent(1));
        for (std::size_t p = 0; p < x.extent(1); ++p)
        {
          double dx = x(0, p) - 0.5;
          double dz = x(2, p) - 0.5;
          double r = std::sqrt(dx * dx + dz * dz);
          v(0, p) = -dz * r * x(1, p);
          v(1, p) = 1.0;
          v(2, p) = dx * r * x(1, p);
        }

        return {vdata, {v.extent(0), v.extent(1)}};
      });

  t0b.stop();

  dolfinx::common::Timer t0c("ZZZ Create forms");

  // Define variational forms
  std::vector form_elasticity_L
      = {form_Elasticity_L1, form_Elasticity_L2, form_Elasticity_L3};
  std::vector form_elasticity_a
      = {form_Elasticity_a1, form_Elasticity_a2, form_Elasticity_a3};
  auto L = std::make_shared<fem::Form<T, double>>(fem::create_form<T>(
      *form_elasticity_L.at(order - 1), {V}, {{"w0", f}}, {}, {}));
  auto a = std::make_shared<fem::Form<T, double>>(fem::create_form<T>(
      *form_elasticity_a.at(order - 1), {V, V},
      std::vector<std::shared_ptr<const fem::Function<T>>>{}, {}, {}));
  t0c.stop();

  dolfinx::common::Timer tassm("ZZZ Assemble matrix");
  dolfinx::common::Timer tcre("Trilinos: create sparsity");
  dolfinx::la::SparsityPattern pattern
      = dolfinx::fem::create_sparsity_pattern(*a);
  pattern.finalize();
  auto [sp_edges, sp_offsets] = pattern.graph();

  const int bs = 3;
  const int num_local = pattern.index_map(0)->size_local();
  std::vector<std::size_t> nnz(num_local * bs);
  for (int i = 0; i < num_local; ++i)
    for (int j = 0; j < bs; ++j)
      nnz[i * bs + j] = bs * (sp_offsets[i + 1] - sp_offsets[i]);

  Teuchos::RCP<const Teuchos::Comm<int>> comm
      = Teuchos::rcp(new Teuchos::MpiComm<int>(mesh->comm()));
  const std::vector<std::int64_t> global_indices = pattern.column_indices();

  std::vector<std::int64_t> global_index_view(global_indices.size() * bs);
  for (std::size_t i = 0; i < global_indices.size(); ++i)
    for (int j = 0; j < bs; ++j)
      global_index_view[i * bs + j] = global_indices[i] * bs + j;
  std::vector<std::int64_t> global_index_vec_view(
      global_index_view.begin(),
      global_index_view.begin() + V->dofmap()->index_map->size_local() * bs);

  Teuchos::RCP<const Tpetra::Map<std::int32_t, std::int64_t>> colMap
      = Teuchos::rcp(new Tpetra::Map<std::int32_t, std::int64_t>(
          V->dofmap()->index_map->size_global() * bs, global_index_view, 0,
          comm));
  Teuchos::RCP<const Tpetra::Map<std::int32_t, std::int64_t>> vecMap
      = Teuchos::rcp(new Tpetra::Map<std::int32_t, std::int64_t>(
          V->dofmap()->index_map->size_global() * bs, global_index_vec_view, 0,
          comm));

  Teuchos::ArrayView<std::size_t> _nnz(nnz.data(), nnz.size());
  Teuchos::RCP<Tpetra::CrsGraph<std::int32_t, std::int64_t>> crs_graph(
      new Tpetra::CrsGraph<std::int32_t, std::int64_t>(vecMap, colMap, _nnz));

  const std::int64_t r0 = V->dofmap()->index_map->local_range()[0];
  for (std::size_t i = 0; i != num_local; ++i)
  {
    std::vector<std::int32_t> indices;
    for (int j = 0; j < bs; ++j)
    {
      for (std::int32_t k = sp_offsets[i]; k < sp_offsets[i + 1]; ++k)
        indices.push_back(sp_edges[k] * bs + j);
    }
    Teuchos::ArrayView<std::int32_t> _indices(indices.data(), indices.size());

    for (int j = 0; j < bs; ++j)
      crs_graph->insertLocalIndices(i * bs + j, _indices);
  }

  crs_graph->fillComplete(vecMap, vecMap);
  tcre.stop();

  // Block matrix (bs=3) for 3D Elasticity
  Teuchos::RCP<Tpetra::CrsMatrix<T, std::int32_t, std::int64_t>> A_Tpetra
      = Teuchos::rcp(
          new Tpetra::CrsMatrix<T, std::int32_t, std::int64_t>(crs_graph));

  // Insert block
  auto tpetra_insert_block = [&A_Tpetra, &bs, &num_local, &global_indices](
                                 const std::span<const std::int32_t>& rows,
                                 const std::span<const std::int32_t>& cols,
                                 const std::span<const T>& data)
  {
    const std::size_t nc = cols.size();
    const std::size_t nr = rows.size();
    std::vector<std::int32_t> col_view(nc * bs);
    for (int k = 0; k < nc; ++k)
      for (int j = 0; j < bs; ++j)
        col_view[k * bs + j] = cols[k] * bs + j;
    for (std::int32_t i = 0; i < nr; ++i)
    {
      if (rows[i] < num_local)
      {
        for (int j = 0; j < bs; ++j)
        {
          Teuchos::ArrayView<const double> data_view(
              data.data() + (i * bs + j) * nc * bs, nc * bs);
          int nvalid = A_Tpetra->sumIntoLocalValues(rows[i] * bs + j, col_view,
                                                    data_view);
          if (nvalid != nc * bs)
            throw std::runtime_error("L Inserted " + std::to_string(nvalid)
                                     + "/" + std::to_string(nc)
                                     + " on row:" + std::to_string(rows[i]));
        }
      }
      else
      {
        std::vector<std::int64_t> global_col_view(nc * bs);
        for (int k = 0; k < nc; ++k)
          for (int j = 0; j < bs; ++j)
            global_col_view[k * bs + j] = global_indices[cols[k]] * bs + j;
        for (int j = 0; j < bs; ++j)
        {
          Teuchos::ArrayView<const double> data_view(
              data.data() + (i * bs + j) * nc * bs, nc * bs);
          int nvalid = A_Tpetra->sumIntoGlobalValues(
              global_indices[rows[i]] * bs + j, global_col_view, data_view);
          if (nvalid != nc * bs)
            throw std::runtime_error("G Inserted " + std::to_string(nvalid)
                                     + "/" + std::to_string(nc)
                                     + " on row:" + std::to_string(rows[i]));
        }
      }
    }
    return 0;
  };

  // Insert individual values (for diagonal)
  std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                    const std::int32_t*, const T*)>
      tpetra_insert
      = [&A_Tpetra, &bs, &num_local, &global_index_view](
            std::int32_t nr, const std::int32_t* rows, const std::int32_t nc,
            const std::int32_t* cols, const T* data)
  {
    std::vector<std::int32_t> col_view(cols, cols + nc);
    for (int i = 0; i < nr; ++i)
    {
      Teuchos::ArrayView<const double> data_view(data + i * nc, nc);
      if (rows[i] < num_local * bs)
      {

        int nvalid = A_Tpetra->sumIntoLocalValues(rows[i], col_view, data_view);

        if (nvalid != nc)
          throw std::runtime_error("LD Inserted " + std::to_string(nvalid) + "/"
                                   + std::to_string(nc)
                                   + " on row:" + std::to_string(rows[i]));
      }
      else
      {
        std::vector<std::int64_t> global_col_view(nc);
        for (int j = 0; j < nc; ++j)
          global_col_view[j] = global_index_view[cols[j]];
        int nvalid = A_Tpetra->sumIntoGlobalValues(global_index_view[rows[i]],
                                                   global_col_view, data_view);

        if (nvalid != nc)
          throw std::runtime_error("GD Inserted " + std::to_string(nvalid) + "/"
                                   + std::to_string(nc)
                                   + " on row:" + std::to_string(rows[i]));
      }
    }

    return 0;
  };

  auto tpetra_set = [&A_Tpetra, &global_indices, &bs,
                     &num_local](const std::span<const std::int32_t>& rows,
                                 const std::span<const std::int32_t>& cols,
                                 const std::span<const T>& data)
  {
    const std::size_t nr = rows.size();
    const std::size_t nc = cols.size();
    if (rows[0] >= (num_local * bs) or nr > 1 or nc > 1)
      throw std::runtime_error("Error setting diagonal: " + std::to_string(nr)
                               + " " + std::to_string(nc) + " "
                               + std::to_string(rows[0]) + "/"
                               + std::to_string(num_local));
    Teuchos::ArrayView<const int> col_view(cols.data(), 1);
    Teuchos::ArrayView<const double> data_view(data.data(), 1);
    int nvalid = A_Tpetra->replaceLocalValues(rows[0], col_view, data_view);
    if (nvalid != nc)
      throw std::runtime_error("Inserted " + std::to_string(nvalid) + "/"
                               + std::to_string(nc) + " on row:"
                               + std::to_string(global_indices[rows[0]]));

    return 0;
  };

  dolfinx::fem::assemble_matrix<T>(tpetra_insert_block, *a, {bc});
  dolfinx::fem::set_diagonal<T>(tpetra_set, *V, {bc});
  A_Tpetra->fillComplete();
  tassm.stop();

  using MV = Tpetra::MultiVector<T, std::int32_t, std::int64_t>;
  using OP = Tpetra::Operator<T, std::int32_t, std::int64_t>;

  la::Vector<T> b(L->function_spaces()[0]->dofmap()->index_map,
                  L->function_spaces()[0]->dofmap()->index_map_bs());
  b.set(0);
  common::Timer t3("ZZZ Assemble vector");
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

  // Create Function to hold solution
  auto u = std::make_shared<dolfinx::fem::Function<T>>(V);

  std::function<int(dolfinx::fem::Function<T>&, const dolfinx::la::Vector<T>&)>
      solver_function = [A_Tpetra, vecMap](dolfinx::fem::Function<T>& u,
                                           const dolfinx::la::Vector<T>& b)
  {
    // FIXME: how to wrap memory with MultiVector? - this is a copy
    const Teuchos::ArrayView<const T> b_view(b.array().data(),
                                             b.array().size());
    Teuchos::RCP<MV> b_Tpetra(new MV(vecMap, b_view, b.array().size(), 1));

    dolfinx::common::Timer ttri("Trilinos: solve");

    // Muelu preconditioner, to be constructed from a Tpetra Operator
    // or Matrix
    Teuchos::RCP<Teuchos::ParameterList> muelu_paramList(
        new Teuchos::ParameterList);
    muelu_paramList->set("problem: type", "Elasticity-3D");
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

    ttri.stop();

    return num_iters;
  };

  return {std::make_shared<la::Vector<T>>(std::move(b)), u, solver_function};
} // namespace

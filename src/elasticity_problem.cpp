// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include "elasticity_problem.h"
#include "Elasticity.h"
#include <BelosSolverFactory.hpp>
#include <BelosTpetraAdapter.hpp>
#include <Eigen/Dense>
#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/array2d.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/la/VectorSpaceBasis.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <utility>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

#include <MueLu_CreateTpetraPreconditioner.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>

namespace
{
// Function to compute the near nullspace for elasticity - it is made up
// of the six rigid body modes
dolfinx::la::VectorSpaceBasis
build_near_nullspace(const dolfinx::fem::FunctionSpace& V)
{
  // Create vectors for nullspace basis
  auto map = V.dofmap()->index_map;
  int bs = V.dofmap()->index_map_bs();
  const std::int32_t length_block = map->size_local() + map->num_ghosts();
  const std::int32_t length = bs * length_block;
  Eigen::Matrix<PetscScalar, Eigen::Dynamic, 6> basis
      = Eigen::Matrix<PetscScalar, Eigen::Dynamic, 6>::Zero(length, 6);

  // NOTE: The below will be simpler once Eigen 3.4 is released, see
  //
  // http://eigen.tuxfamily.org/dox-devel/group__TutorialSlicingIndexing.html

  // x0, x1, x2 translations
  for (int k = 0; k < 3; ++k)
  {
    for (std::int32_t i = 0; i < length_block; ++i)
      basis(bs * i + k, k) = 1.0;
  }

  // Rotations
  const xt::xtensor<double, 2> x = V.tabulate_dof_coordinates(false);
  auto& dofs = V.dofmap()->list().array();
  for (int i = 0; i < dofs.size(); ++i)
  {
    basis(bs * dofs[i] + 0, 3) = -x(dofs[i], 1);
    basis(bs * dofs[i] + 1, 3) = x(dofs[i], 0);

    basis(bs * dofs[i] + 0, 4) = x(dofs[i], 2);
    basis(bs * dofs[i] + 2, 4) = -x(dofs[i], 0);

    basis(bs * dofs[i] + 2, 5) = x(dofs[i], 1);
    basis(bs * dofs[i] + 1, 5) = -x(dofs[i], 2);
  }

  const std::int32_t size = map->size_local() * bs;
  const std::int64_t size_global = map->size_global() * bs;
  std::vector<std::shared_ptr<dolfinx::la::PETScVector>> basis_vec;
  for (int i = 0; i < 6; ++i)
  {
    Vec vec0, vec1;
    VecCreateMPIWithArray(V.mesh()->mpi_comm(), 3, size, size_global,
                          basis.col(i).data(), &vec0);
    VecDuplicate(vec0, &vec1);
    VecCopy(vec0, vec1);
    VecDestroy(&vec0);
    basis_vec.push_back(
        std::make_shared<dolfinx::la::PETScVector>(vec1, false));
  }

  // Create vector space and orthonormalize
  dolfinx::la::VectorSpaceBasis vector_space(basis_vec);
  vector_space.orthonormalize();
  if (!vector_space.is_orthonormal())
    throw std::runtime_error("Space not orthonormal");
  return vector_space;
}
} // namespace

std::tuple<dolfinx::la::PETScMatrix, dolfinx::la::PETScVector,
           std::shared_ptr<dolfinx::fem::Function<PetscScalar>>>
elastic::problem(std::shared_ptr<dolfinx::mesh::Mesh> mesh)
{
  dolfinx::common::Timer t0("ZZZ FunctionSpace");

  auto V = dolfinx::fem::create_functionspace(
      create_functionspace_form_Elasticity_a, "u", mesh);

  t0.stop();

  dolfinx::common::Timer t0a("ZZZ Create boundary conditions");

  // Define boundary condition
  auto u0 = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  std::fill(u0->x()->mutable_array().begin(), u0->x()->mutable_array().end(),
            0.0);

  const std::vector<std::int32_t> bdofs = dolfinx::fem::locate_dofs_geometrical(
      {*V}, [](const xt::xtensor<double, 2>& x) -> xt::xtensor<bool, 1> {
        return xt::isclose(xt::row(x, 1), 0.0);
      });

  // Bottom (x[1] = 0) surface
  auto bc = std::make_shared<dolfinx::fem::DirichletBC<PetscScalar>>(u0, bdofs);

  t0a.stop();

  dolfinx::common::Timer t0b("ZZZ Create RHS function");

  // Define coefficients
  auto f = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  f->interpolate([](const xt::xtensor<double, 2>& x) {
    xt::xtensor<PetscScalar, 2> values({3, x.shape(1)});
    for (std::size_t i = 0; i < x.shape(1); i++)
    {
      double dx = x(0, i) - 0.5;
      double dz = x(2, i) - 0.5;
      double r = dx * dx + dz * dz;
      values(0, i) = -dz * std::sqrt(r) * x(1, i);
      values(1, i) = 1.0;
      values(2, i) = dx * std::sqrt(r) * x(1, i);
    }
    return values;
  });

  t0b.stop();

  dolfinx::common::Timer t0c("ZZZ Create forms");

  // Define variational forms
  auto L = dolfinx::fem::create_form<PetscScalar>(create_form_Elasticity_L, {V},
                                                  {{"f", f}}, {}, {});
  auto a = dolfinx::fem::create_form<PetscScalar>(create_form_Elasticity_a,
                                                  {V, V}, {}, {}, {});
  t0c.stop();

  dolfinx::common::Timer tcre("Trilinos: create sparsity");
  dolfinx::la::SparsityPattern pattern
      = dolfinx::fem::create_sparsity_pattern(*a);
  pattern.assemble();
  const dolfinx::graph::AdjacencyList<std::int32_t>& diagonal_pattern
      = pattern.diagonal_pattern();
  const dolfinx::graph::AdjacencyList<std::int32_t>& off_diagonal_pattern
      = pattern.off_diagonal_pattern();

  const int bs = 3;
  std::vector<std::size_t> nnz(diagonal_pattern.num_nodes() * bs);
  for (int i = 0; i < diagonal_pattern.num_nodes(); ++i)
    for (int j = 0; j < bs; ++j)
      nnz[i * bs + j] = bs
                        * (diagonal_pattern.num_links(i)
                           + off_diagonal_pattern.num_links(i));

  Teuchos::RCP<const Teuchos::Comm<int>> comm
      = Teuchos::rcp(new Teuchos::MpiComm<int>(mesh->mpi_comm()));
  const std::vector<std::int64_t> global_indices
      = pattern.column_indices();

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
  for (std::size_t i = 0; i != diagonal_pattern.num_nodes(); ++i)
  {
    std::vector<std::int32_t> indices;
    for (int j = 0; j < bs; ++j)
    {
      for (std::int32_t col : diagonal_pattern.links(i))
        indices.push_back(col * bs + j);
      for (std::int32_t col : off_diagonal_pattern.links(i))
        indices.push_back(col * bs + j);
    }
    Teuchos::ArrayView<std::int32_t> _indices(indices.data(), indices.size());

    for (int j = 0; j < bs; ++j)
      crs_graph->insertLocalIndices(i * bs + j, _indices);
  }

  crs_graph->fillComplete(vecMap, vecMap);
  tcre.stop();

  // Block matrix (bs=3) for 3D Elasticity
  Teuchos::RCP<Tpetra::CrsMatrix<PetscScalar, std::int32_t, std::int64_t>>
      A_Tpetra = Teuchos::rcp(
          new Tpetra::CrsMatrix<PetscScalar, std::int32_t, std::int64_t>(
              crs_graph));

  const int num_local = V->dofmap()->index_map->size_local();
  // Insert block
  std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                    const std::int32_t*, const PetscScalar*)>
      tpetra_insert_block = [&A_Tpetra, &bs, &num_local, &global_indices](
                                std::int32_t nr, const std::int32_t* rows,
                                const std::int32_t nc, const std::int32_t* cols,
                                const PetscScalar* data) {
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
                  data + (i * bs + j) * nc * bs, nc * bs);
              int nvalid = A_Tpetra->sumIntoLocalValues(rows[i] * bs + j,
                                                        col_view, data_view);
              if (nvalid != nc * bs)
                throw std::runtime_error("L Inserted " + std::to_string(nvalid)
                                         + "/" + std::to_string(nc) + " on row:"
                                         + std::to_string(rows[i]));
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
                  data + (i * bs + j) * nc * bs, nc * bs);
              int nvalid = A_Tpetra->sumIntoGlobalValues(
                  global_indices[rows[i]] * bs + j, global_col_view, data_view);
              if (nvalid != nc * bs)
                throw std::runtime_error("G Inserted " + std::to_string(nvalid)
                                         + "/" + std::to_string(nc) + " on row:"
                                         + std::to_string(rows[i]));
            }
          }
        }
        return 0;
      };

  // Insert individual values (for diagonal)
  std::function<int(std::int32_t, const std::int32_t*, std::int32_t,
                    const std::int32_t*, const PetscScalar*)>
      tpetra_insert = [&A_Tpetra, &bs, &num_local, &global_index_view](
                          std::int32_t nr, const std::int32_t* rows,
                          const std::int32_t nc, const std::int32_t* cols,
                          const PetscScalar* data) {
        std::vector<std::int32_t> col_view(cols, cols + nc);
        for (int i = 0; i < nr; ++i)
        {
          Teuchos::ArrayView<const double> data_view(data + i * nc, nc);
          if (rows[i] < num_local * bs)
          {

            int nvalid
                = A_Tpetra->sumIntoLocalValues(rows[i], col_view, data_view);

            if (nvalid != nc)
              throw std::runtime_error("LD Inserted " + std::to_string(nvalid)
                                       + "/" + std::to_string(nc)
                                       + " on row:" + std::to_string(rows[i]));
          }
          else
          {
            std::vector<std::int64_t> global_col_view(nc);
            for (int j = 0; j < nc; ++j)
              global_col_view[j] = global_index_view[cols[j]];
            int nvalid = A_Tpetra->sumIntoGlobalValues(
                global_index_view[rows[i]], global_col_view, data_view);

            if (nvalid != nc)
              throw std::runtime_error("GD Inserted " + std::to_string(nvalid)
                                       + "/" + std::to_string(nc)
                                       + " on row:" + std::to_string(rows[i]));
          }
        }

        return 0;
      };

  dolfinx::common::Timer tassm("Trilinos: assemble matrix");
  dolfinx::fem::assemble_matrix(tpetra_insert_block, *a, {bc});
  dolfinx::fem::add_diagonal(tpetra_insert, *V, {bc});

  A_Tpetra->fillComplete();
  tassm.stop();

  std::stringstream s;

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
    std::fill(b_.begin(), b_.end(), 0.0);

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
  muelu_paramList->set("problem: type", "Elasticity-3D");
  Teuchos::RCP<MueLu::TpetraOperator<PetscScalar, std::int32_t, std::int64_t>>
      muelu_prec = MueLu::CreateTpetraPreconditioner(
          Teuchos::rcp_dynamic_cast<
              Tpetra::Operator<PetscScalar, std::int32_t, std::int64_t>>(
              A_Tpetra),
          *muelu_paramList);

  // Teuchos::RCP<MV> ns(new MV(vecMap, 6));
  // for (int i = 0; i < 6; ++i)
  // {
  //   const PetscScalar* data;
  //   VecGetArrayRead(nullspace[i]->vec(), &data);
  //   Teuchos::ArrayRCP<PetscScalar> ns_view = ns->getDataNonConst(i);
  //   std::copy(data, data + global_index_vec_view.size(), ns_view.get());
  //   VecRestoreArrayRead(nullspace[i]->vec(), &data);
  // }
  // auto Fine = muelu_prec->GetHierarchy()->GetLevel(0);
  // Fine->setDefaultVerbLevel(Teuchos::VERB_HIGH);
  // Fine->Set("Nullspace", ns);

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

  dolfinx::common::Timer t2("ZZZ Assemble matrix");
  dolfinx::fem::assemble_matrix(dolfinx::la::PETScMatrix::add_block_fn(A.mat()),
                                *a, {bc});
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
  if (dolfinx::MPI::rank(mesh->mpi_comm()) == 0)
    s << "Norm[b](Petsc) = " << norm << "\n";

  std::cout << s.str() << "\n";

  dolfinx::common::Timer t4("ZZZ Create near-nullspace");

  // Create Function to hold solution
  auto u = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);

  // Build near-nullspace and attach to matrix
  dolfinx::la::VectorSpaceBasis nullspace = build_near_nullspace(*V);
  A.set_near_nullspace(nullspace);

  t4.stop();

  return {std::move(A), std::move(b), u};
}

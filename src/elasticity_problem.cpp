// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include "elasticity_problem.h"
#include "Elasticity.h"
#include <Eigen/Dense>
#include <dolfinx/common/Timer.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/la/VectorSpaceBasis.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <utility>

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
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> x
      = V.tabulate_dof_coordinates();
  auto& dofs = V.dofmap()->list().array();
  for (int i = 0; i < dofs.size(); ++i)
  {
    basis.col(3)(bs * dofs[i] + 0) = -x(dofs[i], 1);
    basis.col(3)(bs * dofs[i] + 1) = x(dofs[i], 0);

    basis.col(4)(bs * dofs[i] + 0) = x(dofs[i], 2);
    basis.col(4)(bs * dofs[i] + 2) = -x(dofs[i], 0);

    basis.col(5)(bs * dofs[i] + 2) = x(dofs[i], 1);
    basis.col(5)(bs * dofs[i] + 1) = -x(dofs[i], 2);
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
      {*V}, [](auto& x) { return x.row(1) < 1.0e-8; });

  // Bottom (x[1] = 0) surface
  auto bc = std::make_shared<dolfinx::fem::DirichletBC<PetscScalar>>(u0, bdofs);

  t0a.stop();

  dolfinx::common::Timer t0b("ZZZ Create RHS function");

  // Define coefficients
  auto f = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  f->interpolate([](auto& x) {
    auto dx = x.row(0) - 0.5;
    auto dz = x.row(2) - 0.5;
    auto r = dx * dx + dz * dz;
    Eigen::Array<double, 3, Eigen::Dynamic, Eigen::RowMajor> values(3,
                                                                    x.cols());
    values.row(0) = -dz * r.sqrt() * x.row(1);
    values.row(1) = 1.0;
    values.row(2) = dx * r.sqrt() * x.row(1);
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

  dolfinx::common::Timer t4("ZZZ Create near-nullspace");

  // Create Function to hold solution
  auto u = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);

  // Build near-nullspace and attach to matrix
  dolfinx::la::VectorSpaceBasis nullspace = build_near_nullspace(*V);
  A.set_near_nullspace(nullspace);

  t4.stop();

  return {std::move(A), std::move(b), u};
}

// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include "elasticity_problem.h"
#include "Elasticity.h"
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/array2d.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/la/PETScKrylovSolver.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/PETScVector.h>
#include <dolfinx/la/VectorSpaceBasis.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <utility>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>

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
  xt::xtensor<PetscScalar, 2> basis = xt::zeros<PetscScalar>({length, 6});

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
    xt::xarray<PetscScalar> basis_row = xt::col(basis, i);
    VecCreateMPIWithArray(V.mesh()->mpi_comm(), 3, size, size_global,
                          basis_row.data(), &vec0);
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

std::tuple<std::shared_ptr<dolfinx::la::Vector<PetscScalar>>,
           std::shared_ptr<dolfinx::fem::Function<PetscScalar>>,
           std::function<int(dolfinx::fem::Function<PetscScalar>&,
                             const dolfinx::la::Vector<PetscScalar>&)>>
elastic::problem(std::shared_ptr<dolfinx::mesh::Mesh> mesh, int order)
{
  dolfinx::common::Timer t0("ZZZ FunctionSpace");

  std::vector fs_elasticity
      = {functionspace_form_Elasticity_a1, functionspace_form_Elasticity_a2,
         functionspace_form_Elasticity_a3};
  auto V = dolfinx::fem::create_functionspace(*fs_elasticity.at(order - 1),
                                              "v_0", mesh);

  t0.stop();

  dolfinx::common::Timer t0a("ZZZ Create boundary conditions");

  // Define boundary condition
  auto u0 = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  std::fill(u0->x()->mutable_array().begin(), u0->x()->mutable_array().end(),
            0.0);

  const int tdim = mesh->topology().dim();

  // Find facets with bc applied
  const std::vector<std::int32_t> bc_facets = dolfinx::mesh::locate_entities(
      *mesh, tdim - 1,
      [](const xt::xtensor<double, 2>& x) -> xt::xtensor<bool, 1> {
        return xt::isclose(xt::row(x, 1), 0.0);
      });

  // Find constrained dofs
  const std::vector<std::int32_t> bdofs
      = dolfinx::fem::locate_dofs_topological(*V, tdim - 1, bc_facets);

  // Bottom (x[1] = 0) surface
  auto bc = std::make_shared<dolfinx::fem::DirichletBC<PetscScalar>>(u0, bdofs);

  t0a.stop();

  dolfinx::common::Timer t0b("ZZZ Create RHS function");

  // Define coefficients
  auto f = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);
  f->interpolate(
      [](const xt::xtensor<double, 2>& x)
      {
        xt::xtensor<PetscScalar, 2> values(x.shape());
        auto dx = xt::row(x, 0) - 0.5;
        auto dz = xt::row(x, 2) - 0.5;
        auto r = xt::sqrt(dx * dx + dz * dz);
        xt::row(values, 0) = -dz * r * xt::row(x, 1);
        xt::row(values, 1) = 1.0;
        xt::row(values, 2) = dx * r * xt::row(x, 1);
        return values;
      });

  t0b.stop();

  dolfinx::common::Timer t0c("ZZZ Create forms");

  // Define variational forms
  std::vector form_elasticity_L
      = {form_Elasticity_L1, form_Elasticity_L2, form_Elasticity_L3};
  std::vector form_elasticity_a
      = {form_Elasticity_a1, form_Elasticity_a2, form_Elasticity_a3};
  auto L = std::make_shared<dolfinx::fem::Form<PetscScalar>>(
      dolfinx::fem::create_form<PetscScalar>(*form_elasticity_L.at(order - 1),
                                             {V}, {{"w0", f}}, {}, {}));
  auto a = std::make_shared<
      dolfinx::fem::Form<PetscScalar>>(dolfinx::fem::create_form<PetscScalar>(
      *form_elasticity_a.at(order - 1), {V, V},
      std::vector<std::shared_ptr<const dolfinx::fem::Function<PetscScalar>>>{},
      {}, {}));
  t0c.stop();

  // Create matrices and vector, and assemble system
  std::shared_ptr<dolfinx::la::PETScMatrix> A
      = std::make_shared<dolfinx::la::PETScMatrix>(
          dolfinx::fem::create_matrix(*a), false);

  // Wrap la::Vector with Petsc Vec
  dolfinx::la::Vector<PetscScalar> bx(
      L->function_spaces()[0]->dofmap()->index_map,
      L->function_spaces()[0]->dofmap()->index_map_bs());
  Vec b_vec = dolfinx::la::create_ghosted_vector(
      *(bx.map()), bx.bs(), tcb::span<PetscScalar>(bx.mutable_array()));
  dolfinx::la::PETScVector b(b_vec, false);

  dolfinx::common::Timer t2("ZZZ Assemble matrix");
  const std::vector constants_a = dolfinx::fem::pack_constants(*a);
  const dolfinx::array2d coeffs_a = dolfinx::fem::pack_coefficients(*a);
  dolfinx::fem::assemble_matrix(
      dolfinx::la::PETScMatrix::set_block_fn(A->mat(), ADD_VALUES), *a,
      tcb::make_span(constants_a), coeffs_a, {bc});
  MatAssemblyBegin(A->mat(), MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(A->mat(), MAT_FLUSH_ASSEMBLY);
  dolfinx::fem::set_diagonal(
      dolfinx::la::PETScMatrix::set_fn(A->mat(), INSERT_VALUES), *V, {bc});
  MatAssemblyBegin(A->mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A->mat(), MAT_FINAL_ASSEMBLY);
  t2.stop();

  // Zero PETSc vector
  Vec b_local;
  VecGhostGetLocalForm(b.vec(), &b_local);
  PetscInt n = 0;
  VecGetSize(b_local, &n);
  PetscScalar* array = nullptr;
  VecGetArray(b_local, &array);
  VecSet(b_local, 0.0);
  VecRestoreArray(b_local, &array);
  VecGhostRestoreLocalForm(b.vec(), &b_local);

  dolfinx::common::Timer t3("ZZZ Assemble vector");
  const std::vector constants_L = dolfinx::fem::pack_constants(*L);
  const dolfinx::array2d coeffs_L = dolfinx::fem::pack_coefficients(*L);
  dolfinx::fem::assemble_vector_petsc(b.vec(), *L, constants_L, coeffs_L);
  dolfinx::fem::apply_lifting_petsc(b.vec(), {a}, {constants_L},
                                    {&coeffs_L}, {{bc}}, {}, 1.0);
  VecGhostUpdateBegin(b.vec(), ADD_VALUES, SCATTER_REVERSE);
  VecGhostUpdateEnd(b.vec(), ADD_VALUES, SCATTER_REVERSE);
  dolfinx::fem::set_bc_petsc(b.vec(), {bc}, nullptr);
  t3.stop();

  dolfinx::common::Timer t4("ZZZ Create near-nullspace");

  // Create Function to hold solution
  auto u = std::make_shared<dolfinx::fem::Function<PetscScalar>>(V);

  // Build near-nullspace and attach to matrix
  dolfinx::la::VectorSpaceBasis nullspace = build_near_nullspace(*V);
  A->set_near_nullspace(nullspace);

  t4.stop();

  std::function<int(dolfinx::fem::Function<PetscScalar>&,
                    const dolfinx::la::Vector<PetscScalar>&)>
      solver_function = [A](dolfinx::fem::Function<PetscScalar>& u,
                            const dolfinx::la::Vector<PetscScalar>& b)
  {
    // Create solver
    dolfinx::la::PETScKrylovSolver solver(MPI_COMM_WORLD);
    solver.set_from_options();
    solver.set_operator(A->mat());

    // Wrap dolfinx::la::Vector
    dolfinx::la::Vector<PetscScalar>& bnc
        = const_cast<dolfinx::la::Vector<PetscScalar>&>(b);
    Vec b_petsc = dolfinx::la::create_ghosted_vector(
        *(b.map()), b.bs(), tcb::span<PetscScalar>(bnc.mutable_array()));

    // Solve
    int num_iter = solver.solve(u.vector(), b_petsc);
    return num_iter;
  };

  return {std::make_shared<dolfinx::la::Vector<PetscScalar>>(std::move(bx)), u,
          solver_function};
}

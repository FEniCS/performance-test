// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include "elasticity_problem.h"
#include "Elasticity.h"
#include <dolfinx/common/Timer.h>
#include <dolfinx/fem/DirichletBC.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/assembler.h>
#include <dolfinx/fem/petsc.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/la/petsc.h>
#include <dolfinx/la/utils.h>
#include <dolfinx/mesh/Geometry.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/utils.h>
#include <memory>
#include <petscsys.h>
#include <utility>
#include <xtensor/xarray.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;

namespace
{
// Function to compute the near nullspace for elasticity - it is made up
// of the six rigid body modes
MatNullSpace build_near_nullspace(const fem::FunctionSpace& V)
{
  // Create vectors for nullspace basis
  auto map = V.dofmap()->index_map;
  int bs = V.dofmap()->index_map_bs();
  std::vector<la::Vector<PetscScalar>> basis(6,
                                             la::Vector<PetscScalar>(map, bs));

  // x0, x1, x2 translations
  std::int32_t length_block = map->size_local() + map->num_ghosts();
  for (int k = 0; k < 3; ++k)
  {
    xtl::span<PetscScalar> x = basis[k].mutable_array();
    for (std::int32_t i = 0; i < length_block; ++i)
      x[bs * i + k] = 1.0;
  }

  // Rotations
  auto x3 = basis[3].mutable_array();
  auto x4 = basis[4].mutable_array();
  auto x5 = basis[5].mutable_array();

  // const xt::xtensor<double, 2> x = V.tabulate_dof_coordinates(false);
  const std::vector<double> x = V.tabulate_dof_coordinates(false);
  auto& dofs = V.dofmap()->list().array();
  for (int i = 0; i < dofs.size(); ++i)
  {
    xtl::span<const double, 3> xd(x.data() + 3 * dofs[i], 3);

    x3[bs * dofs[i] + 0] = -xd[1];
    x3[bs * dofs[i] + 1] = xd[0];

    x4[bs * dofs[i] + 0] = xd[2];
    x4[bs * dofs[i] + 2] = -xd[0];

    x5[bs * dofs[i] + 2] = xd[1];
    x5[bs * dofs[i] + 1] = -xd[2];
  }

  // Orthonormalize basis
  la::orthonormalize(tcb::make_span(basis));
  if (!la::is_orthonormal(tcb::span<const decltype(basis)::value_type>(basis)))
  {
    throw std::runtime_error("Space not orthonormal");
  }

  // Build PETSc nullspace object
  std::int32_t length = bs * map->size_local();
  std::vector<xtl::span<const PetscScalar>> basis_local;
  std::transform(basis.cbegin(), basis.cend(), std::back_inserter(basis_local),
                 [length](auto& x)
                 { return xtl::span(x.array().data(), length); });
  MPI_Comm comm = V.mesh()->comm();
  std::vector<Vec> v = la::petsc::create_vectors(comm, basis_local);
  MatNullSpace ns = la::petsc::create_nullspace(comm, v);
  std::for_each(v.begin(), v.end(), [](auto v) { VecDestroy(&v); });
  return ns;
}
} // namespace

std::tuple<std::shared_ptr<la::Vector<PetscScalar>>,
           std::shared_ptr<fem::Function<PetscScalar>>,
           std::function<int(fem::Function<PetscScalar>&,
                             const la::Vector<PetscScalar>&)>>
elastic::problem(std::shared_ptr<mesh::Mesh> mesh, int order)
{
  common::Timer t0("ZZZ FunctionSpace");

  std::vector fs_elasticity
      = {functionspace_form_Elasticity_a1, functionspace_form_Elasticity_a2,
         functionspace_form_Elasticity_a3};
  auto V = std::make_shared<fem::FunctionSpace>(
      fem::create_functionspace(*fs_elasticity.at(order - 1), "v_0", mesh));

  t0.stop();

  common::Timer t0a("ZZZ Create boundary conditions");

  // Define boundary condition
  auto u0 = std::make_shared<fem::Function<PetscScalar>>(V);
  u0->x()->set(0);

  const int tdim = mesh->topology().dim();

  // Find facets with bc applied
  const std::vector<std::int32_t> bc_facets = mesh::locate_entities(
      *mesh, tdim - 1,
      [](const xt::xtensor<double, 2>& x) -> xt::xtensor<bool, 1>
      { return xt::isclose(xt::row(x, 1), 0.0); });

  // Find constrained dofs
  const std::vector<std::int32_t> bdofs
      = fem::locate_dofs_topological(*V, tdim - 1, bc_facets);

  // Bottom (x[1] = 0) surface
  auto bc = std::make_shared<fem::DirichletBC<PetscScalar>>(u0, bdofs);

  t0a.stop();

  common::Timer t0b("ZZZ Create RHS function");

  // Define coefficients
  auto f = std::make_shared<fem::Function<PetscScalar>>(V);
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

  common::Timer t0c("ZZZ Create forms");

  // Define variational forms
  std::vector form_elasticity_L
      = {form_Elasticity_L1, form_Elasticity_L2, form_Elasticity_L3};
  std::vector form_elasticity_a
      = {form_Elasticity_a1, form_Elasticity_a2, form_Elasticity_a3};
  auto L
      = std::make_shared<fem::Form<PetscScalar>>(fem::create_form<PetscScalar>(
          *form_elasticity_L.at(order - 1), {V}, {{"w0", f}}, {}, {}));
  auto a
      = std::make_shared<fem::Form<PetscScalar>>(fem::create_form<PetscScalar>(
          *form_elasticity_a.at(order - 1), {V, V},
          std::vector<std::shared_ptr<const fem::Function<PetscScalar>>>{}, {},
          {}));
  t0c.stop();

  // Create matrices and vector, and assemble system
  std::shared_ptr<la::petsc::Matrix> A = std::make_shared<la::petsc::Matrix>(
      fem::petsc::create_matrix(*a), false);

  common::Timer t2("ZZZ Assemble matrix");
  const std::vector constants_a = fem::pack_constants(*a);
  auto coeffs_a = fem::allocate_coefficient_storage(*a);
  fem::pack_coefficients(*a, coeffs_a);
  fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A->mat(), ADD_VALUES),
                       *a, tcb::make_span(constants_a),
                       fem::make_coefficients_span(coeffs_a), {bc});
  MatAssemblyBegin(A->mat(), MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(A->mat(), MAT_FLUSH_ASSEMBLY);
  fem::set_diagonal<PetscScalar>(
      la::petsc::Matrix::set_fn(A->mat(), INSERT_VALUES), *V, {bc});
  MatAssemblyBegin(A->mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A->mat(), MAT_FINAL_ASSEMBLY);
  t2.stop();

  // Wrap la::Vector with Petsc Vec
  la::Vector<PetscScalar> b(L->function_spaces()[0]->dofmap()->index_map,
                            L->function_spaces()[0]->dofmap()->index_map_bs());
  b.set(0);
  common::Timer t3("ZZZ Assemble vector");
  const std::vector constants_L = fem::pack_constants(*L);
  auto coeffs_L = fem::allocate_coefficient_storage(*L);
  fem::pack_coefficients(*L, coeffs_L);
  fem::assemble_vector<PetscScalar>(b.mutable_array(), *L, constants_L,
                                    fem::make_coefficients_span(coeffs_L));
  fem::apply_lifting(b.mutable_array(), {a}, {constants_L},
                     {fem::make_coefficients_span(coeffs_L)}, {{bc}}, {}, 1.0);
  b.scatter_rev(std::plus<>());
  fem::set_bc(b.mutable_array(), {bc});
  t3.stop();

  common::Timer t4("ZZZ Create near-nullspace");

  // Create Function to hold solution
  auto u = std::make_shared<fem::Function<PetscScalar>>(V);

  // Build near-nullspace and attach to matrix
  MatNullSpace ns = build_near_nullspace(*V);
  MatSetNearNullSpace(A->mat(), ns);
  MatNullSpaceDestroy(&ns);

  t4.stop();

  std::function<int(fem::Function<PetscScalar>&,
                    const la::Vector<PetscScalar>&)>
      solver_function
      = [A](fem::Function<PetscScalar>& u, const la::Vector<PetscScalar>& b)
  {
    // Create solver
    la::petsc::KrylovSolver solver(MPI_COMM_WORLD);
    solver.set_from_options();
    solver.set_operator(A->mat());

    // Wrap la::Vector
    la::petsc::Vector _b(la::petsc::create_vector_wrap(b), false);
    la::petsc::Vector x(la::petsc::create_vector_wrap(*u.x()), false);

    // Solve
    int num_iter = solver.solve(x.vec(), _b.vec());
    return num_iter;
  };

  return {std::make_shared<la::Vector<PetscScalar>>(std::move(b)), u,
          solver_function};
}

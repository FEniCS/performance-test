// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include "crescendo_elasticity_problem.h"
#include "Elasticity.h"
#include <basix/mdspan.hpp>
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
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/utils.h>
#include <memory>
#include <petscsys.h>
#include <petscksp.h> 
#include <petscmat.h>
#include <span>
#include <utility>

using namespace dolfinx;
using T = PetscScalar;

extern "C"
{
  PetscErrorCode fmonitor(KSP ksp, PetscInt i, PetscReal r, void *ctx)
  {
    auto vec0 = (dolfinx::la::Vector<double> *)ctx;

    if (i%10 == 0)
    {
      Vec t, v, vec;
      //      la::petsc::Vector x(la::petsc::create_vector_wrap(*vec0), false);
      //      Vec vec = x.vec();
      KSPBuildResidual(ksp, t, v, &vec);

      double vnorm;
      VecNorm(vec, NORM_2, &vnorm);

      spdlog::info("v.norm = {} {}", vnorm, r);
    }

    return 0;
  }
  

  PetscErrorCode fdestroy(void **ctx)
  {
    return 0;
  }
  
}


namespace
{
// Function to compute the near nullspace for elasticity - it is made up
// of the six rigid body modes
MatNullSpace build_near_nullspace(const fem::FunctionSpace<double>& V)
{
  // Create vectors for nullspace basis
  auto map = V.dofmap()->index_map;
  int bs = V.dofmap()->index_map_bs();
  std::vector<la::Vector<T>> basis(6, la::Vector<T>(map, bs));

  // x0, x1, x2 translations
  std::int32_t length_block = map->size_local() + map->num_ghosts();
  for (int k = 0; k < 3; ++k)
  {
    std::span<T> x = basis[k].mutable_array();
    for (std::int32_t i = 0; i < length_block; ++i)
      x[bs * i + k] = 1.0;
  }

  // Rotations
  auto x3 = basis[3].mutable_array();
  auto x4 = basis[4].mutable_array();
  auto x5 = basis[5].mutable_array();

  const std::vector<double> x = V.tabulate_dof_coordinates(false);
  const std::int32_t* dofs = V.dofmap()->map().data_handle();
  for (std::size_t i = 0; i < V.dofmap()->map().size(); ++i)
  {
    std::span<const double, 3> xd(x.data() + 3 * dofs[i], 3);

    x3[bs * dofs[i] + 0] = -xd[1];
    x3[bs * dofs[i] + 1] = xd[0];

    x4[bs * dofs[i] + 0] = xd[2];
    x4[bs * dofs[i] + 2] = -xd[0];

    x5[bs * dofs[i] + 2] = xd[1];
    x5[bs * dofs[i] + 1] = -xd[2];
  }

  // Orthonormalize basis
  la::orthonormalize(std::vector<std::reference_wrapper<la::Vector<T>>>(
      basis.begin(), basis.end()));
  if (!la::is_orthonormal(
          std::vector<std::reference_wrapper<const la::Vector<T>>>(
              basis.begin(), basis.end())))
  {
    throw std::runtime_error("Space not orthonormal");
  }

  // Build PETSc nullspace object
  std::int32_t length = bs * map->size_local();
  std::vector<std::span<const T>> basis_local;
  std::transform(basis.cbegin(), basis.cend(), std::back_inserter(basis_local),
                 [length](auto& x)
                 { return std::span(x.array().data(), length); });
  MPI_Comm comm = V.mesh()->comm();
  std::vector<Vec> v = la::petsc::create_vectors(comm, basis_local);
  MatNullSpace ns = la::petsc::create_nullspace(comm, v);
  std::for_each(v.begin(), v.end(), [](auto v) { VecDestroy(&v); });
  return ns;
}
} // namespace

std::tuple<std::shared_ptr<la::Vector<T>>, std::shared_ptr<fem::Function<T>>,
           std::function<int(fem::Function<T>&, const la::Vector<T>&)>>
crescendo_elastic::problem(std::shared_ptr<mesh::Mesh<double>> mesh, std::shared_ptr<dolfinx::mesh::MeshTags<std::int32_t>> facet_markers)
{
  common::Timer t0("ZZZ FunctionSpace");

  constexpr int order = 1;
  
  auto element = basix::create_element<double>(
      basix::element::family::P, basix::cell::type::tetrahedron, order,
      basix::element::lagrange_variant::gll_warped,
      basix::element::dpc_variant::unset, false);

  auto V = std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace(mesh, element, {3}));

  t0.stop();

  common::Timer t0a("ZZZ Create boundary conditions");

  // Define boundary condition
  auto u0 = std::make_shared<fem::Function<T>>(V);
  u0->x()->set(0);
  auto u1 = std::make_shared<fem::Function<T>>(V);
  spdlog::info("u1 array size = {}", u1->x()->array().size());
  auto u1array = u1->x()->mutable_array();
  for (std::size_t i = 0; i < u1array.size(); i += 3)
  {
    u1array[i] = 1.0;
    u1array[i + 1] = 0.0;
    u1array[i + 2] = 0.0;
  }
  
  const int tdim = mesh->topology()->dim();

  // Find facets with bc applied
  const std::vector<std::int32_t> bc_facets0 = facet_markers->find(144);
  std::vector<std::int32_t> bc_facets1 = facet_markers->find(59);
  //  std::vector<std::int32_t> bc_facets2 = facet_markers->find(70);
  //  bc_facets1.insert(bc_facets1.end(), bc_facets2.begin(), bc_facets2.end());

  spdlog::info("Facets 0 (144) = {}", bc_facets0.size());
  spdlog::info("Facets 1 (59) = {}", bc_facets1.size());
  
  
  // Find constrained dofs
  const std::vector<std::int32_t> bdofs0 = fem::locate_dofs_topological(
      *V->mesh()->topology_mutable(), *V->dofmap(), tdim - 1, bc_facets0);

  const std::vector<std::int32_t> bdofs1 = fem::locate_dofs_topological(
      *V->mesh()->topology_mutable(), *V->dofmap(), tdim - 1, bc_facets1);

  // Constrained surface
  auto bc0 = std::make_shared<fem::DirichletBC<T>>(u0, bdofs0);
  auto bc1 = std::make_shared<fem::DirichletBC<T>>(u1, bdofs1);

  t0a.stop();

  common::Timer t0b("ZZZ Create RHS function");

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
          v(0, p) = 0.0 * -dz * r * x(1, p);
          v(1, p) = 0.0;
          v(2, p) = 0.0 * dx * r * x(1, p);
        }

        return {vdata, {v.extent(0), v.extent(1)}};
      });

  t0b.stop();

  common::Timer t0c("ZZZ Create forms");

  // Define variational forms
  std::vector form_elasticity_L
      = {form_Elasticity_L1, form_Elasticity_L2, form_Elasticity_L3};
  std::vector form_elasticity_a
      = {form_Elasticity_a1, form_Elasticity_a2, form_Elasticity_a3};
  auto L = std::make_shared<fem::Form<T, double>>(fem::create_form<T>(
      *form_elasticity_L.at(order - 1), {V}, {{"w0", f}}, {}, {}));
  auto a = std::make_shared<fem::Form<T, double>>(fem::create_form<T>(
      *form_elasticity_a.at(order - 1), {V, V}, {}, {}, {}));
  t0c.stop();

  // Create matrices and vector, and assemble system
  std::shared_ptr<la::petsc::Matrix> A = std::make_shared<la::petsc::Matrix>(
      fem::petsc::create_matrix(*a), false);

  common::Timer t2("ZZZ Assemble matrix");
  const std::vector constants_a = fem::pack_constants(*a);
  auto coeffs_a = fem::allocate_coefficient_storage(*a);
  fem::pack_coefficients(*a, coeffs_a);
  fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A->mat(), ADD_VALUES),
                       *a, std::span(constants_a),
                       fem::make_coefficients_span(coeffs_a), {bc0, bc1});
  MatAssemblyBegin(A->mat(), MAT_FLUSH_ASSEMBLY);
  MatAssemblyEnd(A->mat(), MAT_FLUSH_ASSEMBLY);
  fem::set_diagonal<T>(la::petsc::Matrix::set_fn(A->mat(), INSERT_VALUES), *V,
                       {bc0, bc1});
  MatAssemblyBegin(A->mat(), MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A->mat(), MAT_FINAL_ASSEMBLY);
  t2.stop();

  // Wrap la::Vector with Petsc Vec
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
                                {fem::make_coefficients_span(coeffs_L)}, {{bc0, bc1}},
                                {}, 1.0);
  b.scatter_rev(std::plus<>());
  fem::set_bc<T, double>(b.mutable_array(), {bc0, bc1});
  t3.stop();

  common::Timer t4("ZZZ Create near-nullspace");

  // Create Function to hold solution
  auto u = std::make_shared<fem::Function<T>>(V);

  // Build near-nullspace and attach to matrix
  MatNullSpace ns = build_near_nullspace(*V);
  MatSetNearNullSpace(A->mat(), ns);
  MatNullSpaceDestroy(&ns);

  t4.stop();

  std::function<int(fem::Function<T>&, const la::Vector<T>&)> solver_function
      = [A](fem::Function<T>& u, const la::Vector<T>& b)
  {
    // Create solver
    la::petsc::KrylovSolver solver(MPI_COMM_WORLD);
    solver.set_from_options();
    solver.set_operator(A->mat());


    
    // Wrap la::Vector
    la::petsc::Vector _b(la::petsc::create_vector_wrap(b), false);
    la::petsc::Vector x(la::petsc::create_vector_wrap(*u.x()), false);

    dolfinx::fem::Function<T> resid(u.function_space());
    void *ctx= (void *)(resid.x().get());
    
    KSPMonitorSet(solver.ksp(), fmonitor, ctx, fdestroy);
    
    
    // Solve
    int num_iter = solver.solve(x.vec(), _b.vec());
    return num_iter;
  };

  return {std::make_shared<la::Vector<T>>(std::move(b)), u, solver_function};
}

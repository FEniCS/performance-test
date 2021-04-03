// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include "Elasticity.h"
#include "Poisson.h"
#include "elasticity_problem.h"
#include "mesh.h"
#include "poisson_problem.h"
#include <boost/program_options.hpp>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/subsystem.h>
#include <dolfinx/common/timing.h>
#include <dolfinx/common/version.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/fem/Function.h>
#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/la/PETScKrylovSolver.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/PETScVector.h>
#include <string>
#include <utility>

namespace po = boost::program_options;

void solve(int argc, char* argv[])
{
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "print usage message")(
      "problem_type", po::value<std::string>()->default_value("poisson"),
      "problem (poisson or elasticity)")(
      "mesh_type", po::value<std::string>()->default_value("cube"),
      "mesh (cube or unstructured)")(
      "scaling_type", po::value<std::string>()->default_value("weak"),
      "scaling (weak or strong)")(
      "output", po::value<std::string>()->default_value(""),
      "output directory (no output unless this is set)")(
      "ndofs", po::value<std::size_t>()->default_value(50000),
      "number of degrees of freedom");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .options(desc)
                .allow_unregistered()
                .run(),
            vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << "\n";
    return;
  }

  const std::string problem_type = vm["problem_type"].as<std::string>();
  const std::string mesh_type = vm["mesh_type"].as<std::string>();
  const std::string scaling_type = vm["scaling_type"].as<std::string>();
  const std::size_t ndofs = vm["ndofs"].as<std::size_t>();
  const std::string output_dir = vm["output"].as<std::string>();
  const bool output = (output_dir.size() > 0);

  bool strong_scaling;
  if (scaling_type == "strong")
    strong_scaling = true;
  else if (scaling_type == "weak")
    strong_scaling = false;
  else
    throw std::runtime_error("Scaling type '" + scaling_type + "` unknown");

  // Get number of processes
  const std::size_t num_processes = dolfinx::MPI::size(MPI_COMM_WORLD);

  // Assemble problem
  std::shared_ptr<dolfinx::mesh::Mesh> mesh;
  std::shared_ptr<dolfinx::la::PETScMatrix> A;
  std::shared_ptr<dolfinx::la::PETScVector> b;
  std::shared_ptr<dolfinx::fem::Function<PetscScalar>> u;
  if (problem_type == "poisson")
  {
    dolfinx::common::Timer t0("ZZZ Create Mesh");
    auto cmap
        = dolfinx::fem::create_coordinate_map(create_coordinate_map_Poisson);
    if (mesh_type == "cube")
      mesh = create_cube_mesh(MPI_COMM_WORLD, ndofs, strong_scaling, 1, cmap);
    else
      mesh = create_spoke_mesh(MPI_COMM_WORLD, ndofs, strong_scaling, 1, cmap);
    t0.stop();

    // Create mesh entity permutations outside of the assembler
    dolfinx::common::Timer tperm("ZZZ Create mesh entity permutations");
    mesh->topology_mutable().create_entity_permutations();
    tperm.stop();

    // Create Poisson problem
    auto data = poisson::problem(mesh);
    A = std::make_shared<dolfinx::la::PETScMatrix>(
        std::move(std::get<0>(data)));
    b = std::make_shared<dolfinx::la::PETScVector>(
        std::move(std::get<1>(data)));
    u = std::get<2>(data);
  }
  else if (problem_type == "elasticity")
  {
    dolfinx::common::Timer t0("ZZZ Create Mesh");
    auto cmap
        = dolfinx::fem::create_coordinate_map(create_coordinate_map_Elasticity);
    if (mesh_type == "cube")
      mesh = create_cube_mesh(MPI_COMM_WORLD, ndofs, strong_scaling, 3, cmap);
    else
      mesh = create_spoke_mesh(MPI_COMM_WORLD, ndofs, strong_scaling, 3, cmap);
    t0.stop();

    // Create mesh entity permutations outside of the assembler
    dolfinx::common::Timer tperm("ZZZ Create mesh entity permutations");
    mesh->topology_mutable().create_entity_permutations();
    tperm.stop();

    // Create elasticity problem. Near-nullspace will be attached to the
    // linear operator (matrix).
    auto data = elastic::problem(mesh);
    A = std::make_shared<dolfinx::la::PETScMatrix>(
        std::move(std::get<0>(data)));
    b = std::make_shared<dolfinx::la::PETScVector>(
        std::move(std::get<1>(data)));
    u = std::get<2>(data);
  }
  else
    throw std::runtime_error("Unknown problem type: " + problem_type);

  const std::int32_t num_ghosts
      = u->function_space()->dofmap()->index_map->num_ghosts()
        * u->function_space()->dofmap()->index_map_bs();
  std::int32_t min_ghosts = -1;
  std::int32_t max_ghosts = -1;
  MPI_Reduce(&num_ghosts, &min_ghosts, 1, MPI_INT32_T, MPI_MIN, 0,
             MPI_COMM_WORLD);

  MPI_Reduce(&num_ghosts, &max_ghosts, 1, MPI_INT32_T, MPI_MAX, 0,
             MPI_COMM_WORLD);

  // Print simulation summary
  if (dolfinx::MPI::rank(MPI_COMM_WORLD) == 0)
  {
    char petsc_version[256];
    PetscGetVersion(petsc_version, 256);

    const std::int64_t num_dofs
        = u->function_space()->dofmap()->index_map->size_global()
          * u->function_space()->dofmap()->index_map_bs();
    const int tdim = mesh->topology().dim();
    const std::int64_t num_cells
        = mesh->topology().index_map(tdim)->size_global();

    std::cout
        << "----------------------------------------------------------------"
        << std::endl;
    std::cout << "Test problem summary" << std::endl;
    std::cout << "  dolfinx version: " << DOLFINX_VERSION_STRING << std::endl;
    std::cout << "  dolfinx hash:    " << DOLFINX_VERSION_GIT << std::endl;
    std::cout << "  ufl hash:        " << UFC_SIGNATURE << std::endl;
    std::cout << "  petsc version:   " << petsc_version << std::endl;
    std::cout << "  Problem type:    " << problem_type << std::endl;
    std::cout << "  Scaling type:    " << scaling_type << std::endl;
    std::cout << "  Num processes:   " << num_processes << std::endl;
    std::cout << "  Num cells        " << num_cells << std::endl;
    std::cout << "  Num ghosts min   " << min_ghosts << std::endl;
    std::cout << "  Num ghosts max   " << max_ghosts << std::endl;
    std::cout << "  Total degrees of freedom:               " << num_dofs
              << std::endl;
    std::cout << "  Average degrees of freedom per process: "
              << num_dofs / dolfinx::MPI::size(MPI_COMM_WORLD) << std::endl;

    std::cout
        << "----------------------------------------------------------------"
        << std::endl;
  }

  // Display timings
  dolfinx::list_timings(MPI_COMM_WORLD, {dolfinx::TimingType::wall});
}

int main(int argc, char* argv[])
{
  dolfinx::common::subsystem::init_logging(argc, argv);
  dolfinx::common::subsystem::init_mpi();
  dolfinx::common::subsystem::init_petsc(argc, argv);

  solve(argc, argv);

  std::cout << "\n\n\n\n";

  MPI_Barrier(MPI_COMM_WORLD);
  int mRank, mSize;
  MPI_Comm_rank(MPI_COMM_WORLD, &mRank);
  MPI_Comm_size(MPI_COMM_WORLD, &mSize);

  char name[MPI_MAX_PROCESSOR_NAME];
  int len;
  MPI_Get_processor_name(name, &len);
  printf("Node number %d/%d is %s \n", mRank, mSize, name);

  dolfinx::common::subsystem::finalize_petsc();
  dolfinx::common::subsystem::finalize_mpi();
  return 0;
}

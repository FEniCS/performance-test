// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include "Elasticity.h"
#include "Poisson.h"
#include "elasticity_problem.h"
#include "elasticity_problem_trilinos.h"
#include "mesh.h"
#include "poisson_problem.h"
#include "poisson_problem_trilinos.h"
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
#include <dolfinx/la/Vector.h>
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
  std::shared_ptr<dolfinx::la::Vector<PetscScalar>> b;
  std::shared_ptr<dolfinx::fem::Function<PetscScalar>> u;
  std::function<int(dolfinx::fem::Function<PetscScalar>&,
                    const dolfinx::la::Vector<PetscScalar>&)>
      solver_function;

  int ndofs_per_node = 0;
  std::shared_ptr<dolfinx::fem::CoordinateElement> cmap;
  if (problem_type == "elasticity" or problem_type == "elasticity_trilinos")
  {
    cmap = std::make_shared<dolfinx::fem::CoordinateElement>(
        dolfinx::fem::create_coordinate_map(create_coordinate_map_Elasticity));
    ndofs_per_node = 3;
  }
  else
  {
    cmap = std::make_shared<dolfinx::fem::CoordinateElement>(
        dolfinx::fem::create_coordinate_map(create_coordinate_map_Poisson));
    ndofs_per_node = 1;
  }

  dolfinx::common::Timer t0("ZZZ Create Mesh");
  if (mesh_type == "cube")
    mesh = create_cube_mesh(MPI_COMM_WORLD, ndofs, strong_scaling,
                            ndofs_per_node, *cmap);
  else
    mesh = create_spoke_mesh(MPI_COMM_WORLD, ndofs, strong_scaling,
                             ndofs_per_node, *cmap);
  t0.stop();

  // Create mesh entity permutations outside of the assembler
  dolfinx::common::Timer tperm("ZZZ Create mesh entity permutations");
  mesh->topology_mutable().create_entity_permutations();
  tperm.stop();

  // Create Poisson problem
  if (problem_type == "poisson")
    std::tie(b, u, solver_function) = poisson::problem(mesh);
  else if (problem_type == "poisson_trilinos")
    std::tie(b, u, solver_function) = poisson_trilinos::problem(mesh);
  else if (problem_type == "elasticity")
    std::tie(b, u, solver_function) = elastic::problem(mesh);
  else if (problem_type == "elasticity_trilinos")
    std::tie(b, u, solver_function) = elastic_trilinos::problem(mesh);
  else
    throw std::runtime_error("Unknown problem type: " + problem_type);

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
    std::cout << "  Total degrees of freedom:               " << num_dofs
              << std::endl;
    std::cout << "  Average degrees of freedom per process: "
              << num_dofs / dolfinx::MPI::size(MPI_COMM_WORLD) << std::endl;
    std::cout
        << "----------------------------------------------------------------"
        << std::endl;
  }

  dolfinx::common::Timer t5("ZZZ Solve");
  int num_iter = solver_function(*u, *b);
  t5.stop();

  if (output)
  {
    dolfinx::common::Timer t6("ZZZ Output");
    std::string filename
        = output_dir + "/solution-" + std::to_string(num_processes) + ".xdmf";
    dolfinx::io::XDMFFile file(MPI_COMM_WORLD, filename, "w");
    file.write_mesh(*mesh);
    file.write_function(*u, 0.0);
    t6.stop();
  }

  // Display timings
  dolfinx::list_timings(MPI_COMM_WORLD, {dolfinx::TimingType::wall});

  PetscReal norm = 0.0;
  VecNorm(u->vector(), NORM_2, &norm);
  // Report number of Krylov iterations
  if (dolfinx::MPI::rank(MPI_COMM_WORLD) == 0)
  {
    std::cout << "*** Number of Krylov iterations: " << num_iter << std::endl;
    std::cout << "*** Solution norm:  " << norm << std::endl;
  }
}

int main(int argc, char* argv[])
{
  dolfinx::common::subsystem::init_logging(argc, argv);
  dolfinx::common::subsystem::init_mpi();
  dolfinx::common::subsystem::init_petsc(argc, argv);

  solve(argc, argv);

  dolfinx::common::subsystem::finalize_petsc();
  dolfinx::common::subsystem::finalize_mpi();
  return 0;
}

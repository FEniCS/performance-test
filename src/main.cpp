// Copyright (C) 2017-2022 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include "elasticity_problem.h"
#include "mem.h"
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
#include <dolfinx/la/Vector.h>
#include <iomanip>
#include <petscsys.h>
#include <string>
#include <thread>
#include <utility>

namespace po = boost::program_options;

std::string int64_to_human(std::int64_t n)
{
  double r = static_cast<double>(n);
  const std::string name[] = {"", "thousand", "million", "billion", "trillion"};

  int i = 0;
  while (r > 1000.0)
  {
    r /= 1000.0;
    i++;
  }
  if (i > 4)
    throw std::runtime_error("number too big");

  std::stringstream s;
  if (i == 0)
    return s.str();
  s << " (" << std::setprecision(3) << r << " " << name[i] << ")";
  return s.str();
}

void solve(int argc, char* argv[])
{
  po::options_description desc("Allowed options");
  bool mem_profile;
  desc.add_options()("help,h", "print usage message")(
      "problem_type", po::value<std::string>()->default_value("poisson"),
      "problem (poisson or elasticity)")(
      "mesh_type", po::value<std::string>()->default_value("cube"),
      "mesh (cube or unstructured)")(
      "memory_profiling", po::bool_switch(&mem_profile)->default_value(false),
      "turn on memory logging")("scaling_type",
                                po::value<std::string>()->default_value("weak"),
                                "scaling (weak or strong)")(
      "output", po::value<std::string>()->default_value(""),
      "output directory (no output unless this is set)")(
      "ndofs", po::value<std::size_t>()->default_value(50000),
      "number of degrees of freedom")(
      "max_init_cells_per_dim", po::value<std::size_t>()->default_value(100),
      "maximum number cells in each direction before refinement")(
      "target_cells_per_rank", po::value<int>()->default_value(-1),
      "target number of cells per rank for the graph partitioner")(
      "order", po::value<std::size_t>()->default_value(1), "polynomial order");

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

  const std::size_t max_cells_per_dim
      = vm["max_init_cells_per_dim"].as<std::size_t>();
  const int target_cells_per_rank = vm["target_cells_per_rank"].as<int>();

  const int order = vm["order"].as<std::size_t>();
  const std::string output_dir = vm["output"].as<std::string>();
  const bool output = (output_dir.size() > 0);
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);

  bool quit_flag = false;
  std::thread mem_thread;

  if (mem_profile and mpi_rank == 0)
  {
    mem_thread = std::thread(process_mem_usage, std::ref(quit_flag));
  }

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

  const int ndofs_per_node = (problem_type == "elasticity") ? 3 : 1;

  dolfinx::common::Timer t0("ZZZ Create Mesh");
  if (mesh_type == "cube")
  {
    mesh = std::make_shared<dolfinx::mesh::Mesh>(
        create_cube_mesh(MPI_COMM_WORLD, ndofs, strong_scaling, ndofs_per_node,
                         max_cells_per_dim, target_cells_per_rank, order));
  }
  else
  {
    mesh = create_spoke_mesh(MPI_COMM_WORLD, ndofs, strong_scaling,
                             ndofs_per_node);
  }
  t0.stop();

  dolfinx::common::Timer t_ent(
      "ZZZ Create facets and facet->cell connectivity");
  mesh->topology_mutable().create_entities(2);
  mesh->topology_mutable().create_connectivity(2, 3);
  t_ent.stop();

  if (problem_type == "poisson")
  {
    // Create Poisson problem
    std::tie(b, u, solver_function) = poisson::problem(mesh, order);
  }
  else if (problem_type == "elasticity")
  {
    // Create elasticity problem. Near-nullspace will be attached to the
    // linear operator (matrix).
    std::tie(b, u, solver_function) = elastic::problem(mesh, order);
  }
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
    const std::string num_cells_human = int64_to_human(num_cells);
    const std::string num_dofs_human = int64_to_human(num_dofs);
    std::cout
        << "----------------------------------------------------------------"
        << std::endl;
    std::cout << "Test problem summary" << std::endl;
    std::cout << "  dolfinx version: " << DOLFINX_VERSION_STRING << std::endl;
    std::cout << "  dolfinx hash:    " << DOLFINX_VERSION_GIT << std::endl;
    std::cout << "  ufl hash:        " << UFCX_SIGNATURE << std::endl;
    std::cout << "  petsc version:   " << petsc_version << std::endl;
    std::cout << "  Problem type:    " << problem_type << std::endl;
    std::cout << "  Scaling type:    " << scaling_type << std::endl;
    std::cout << "  Num processes:   " << num_processes << std::endl;
    std::cout << "  Num cells:       " << num_cells << num_cells_human
              << std::endl;
    std::cout << "  Total degrees of freedom:               " << num_dofs
              << num_dofs_human << std::endl;
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

  // Report number of Krylov iterations
  double norm = u->x()->norm();
  if (dolfinx::MPI::rank(MPI_COMM_WORLD) == 0)
  {
    std::cout << "*** Number of Krylov iterations: " << num_iter << std::endl;
    std::cout << "*** Solution norm:  " << norm << std::endl;
  }

  if (mem_profile and mpi_rank == 0)
  {
    quit_flag = true;
    mem_thread.join();
  }
}

int main(int argc, char* argv[])
{
  dolfinx::common::Timer t0("Init MPI");
  dolfinx::common::subsystem::init_mpi();
  t0.stop();
  dolfinx::common::Timer t1("Init logging");
  dolfinx::common::subsystem::init_logging(argc, argv);
  t1.stop();
  dolfinx::common::Timer t2("Init PETSc");
  dolfinx::common::subsystem::init_petsc(argc, argv);
  t2.stop();

  // Set the logging thread name to show the process rank
  // and enable on rank 0 (add more here if desired)
  const int mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);
  if (mpi_rank == 0)
    loguru::g_stderr_verbosity = loguru::Verbosity_INFO;

  std::string thread_name = "RANK: " + std::to_string(mpi_rank);
  loguru::set_thread_name(thread_name.c_str());
  loguru::g_stderr_verbosity = loguru::Verbosity_INFO;

  solve(argc, argv);

  dolfinx::common::subsystem::finalize_petsc();
  dolfinx::common::subsystem::finalize_mpi();
  return 0;
}

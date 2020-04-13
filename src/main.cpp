// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#include "elasticity_problem.h"
#include "mesh.h"
#include "poisson_problem.h"
#include <boost/program_options.hpp>
#include <dolfinx/common/SubSystemsManager.h>
#include <dolfinx/common/Timer.h>
#include <dolfinx/common/timing.h>
#include <dolfinx/fem/Form.h>
#include <dolfinx/function/Function.h>
#include <dolfinx/function/FunctionSpace.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/la/PETScKrylovSolver.h>
#include <dolfinx/la/PETScMatrix.h>
#include <dolfinx/la/PETScVector.h>
#include <string>
#include <utility>

namespace po = boost::program_options;

int main(int argc, char* argv[])
{
  dolfinx::common::SubSystemsManager::init_logging(argc, argv);
  dolfinx::common::SubSystemsManager::init_mpi();
  dolfinx::common::SubSystemsManager::init_petsc(argc, argv);

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
    return 0;
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
  std::shared_ptr<dolfinx::function::Function> u;
  if (problem_type == "poisson")
  {
    dolfinx::common::Timer t0("ZZZ Create Mesh");
    if (mesh_type == "cube")
      mesh = create_cube_mesh(MPI_COMM_WORLD, ndofs, strong_scaling, 1);
    else
      mesh = create_spoke_mesh(MPI_COMM_WORLD, ndofs, strong_scaling, 1);
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
    if (mesh_type == "cube")
      mesh = create_cube_mesh(MPI_COMM_WORLD, ndofs, strong_scaling, 3);
    else
      mesh = create_spoke_mesh(MPI_COMM_WORLD, ndofs, strong_scaling, 3);
    t0.stop();

    // Create mesh entity permutations outside of the assembler
    dolfinx::common::Timer tperm("ZZZ Create mesh entity permutations");
    mesh->create_entity_permutations();
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

  // Print simulation summary
  if (dolfinx::MPI::rank(MPI_COMM_WORLD) == 0)
  {
    std::cout
        << "----------------------------------------------------------------"
        << std::endl;
    std::cout << "Test problem summary" << std::endl;
    std::cout << "  Problem type:   " << problem_type << std::endl;
    std::cout << "  Scaling type:   " << scaling_type << std::endl;
    std::cout << "  Num processes:  " << num_processes << std::endl;
    std::cout << "  Total degrees of freedom:               "
              << u->function_space()->dim() << std::endl;
    std::cout << "  Average degrees of freedom per process: "
              << u->function_space()->dim() / dolfinx::MPI::size(MPI_COMM_WORLD)
              << std::endl;
    std::cout
        << "----------------------------------------------------------------"
        << std::endl;
  }

  // Create solver
  dolfinx::la::PETScKrylovSolver solver(MPI_COMM_WORLD);
  solver.set_from_options();
  solver.set_operator(A->mat());

  // Solve
  dolfinx::common::Timer t5("ZZZ Solve");
  int num_iter = solver.solve(u->vector().vec(), b->vec());

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

  double norm = u->vector().norm(dolfinx::la::Norm::l2);
  // Report number of Krylov iterations
  if (dolfinx::MPI::rank(MPI_COMM_WORLD) == 0)
  {
    std::cout << "*** Number of Krylov iterations: " << num_iter << std::endl;
    std::cout << "*** Solution norm:  " << norm << std::endl;
  }

  return 0;
}

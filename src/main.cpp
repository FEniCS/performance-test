// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
// Licensed under the MIT License. See LICENSE file in the project root
// for full license information.

#include "elasticity_problem.h"
#include "mesh.h"
#include "poisson_problem.h"
#include <boost/program_options.hpp>
#include <dolfin/common/SubSystemsManager.h>
#include <dolfin/common/Timer.h>
#include <dolfin/common/timing.h>
#include <dolfin/fem/Form.h>
#include <dolfin/function/Function.h>
#include <dolfin/function/FunctionSpace.h>
#include <dolfin/io/XDMFFile.h>
#include <dolfin/la/PETScKrylovSolver.h>
#include <dolfin/la/PETScMatrix.h>
#include <dolfin/la/PETScVector.h>
#include <string>
#include <utility>

namespace po = boost::program_options;

int main(int argc, char* argv[])
{
  dolfin::common::SubSystemsManager::init_mpi();
  dolfin::common::SubSystemsManager::init_petsc(argc, argv);

  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "print usage message")(
      "problem_type", po::value<std::string>()->default_value("poisson"),
      "problem (poisson or elasticity)")(
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
  const std::size_t num_processes = dolfin::MPI::size(MPI_COMM_WORLD);

  // Assemble problem
  std::shared_ptr<dolfin::la::PETScMatrix> A;
  std::shared_ptr<dolfin::la::PETScVector> b;
  std::shared_ptr<dolfin::function::Function> u;
  if (problem_type == "poisson")
  {
    dolfin::common::Timer t0("ZZZ Create Mesh");
    std::shared_ptr<dolfin::mesh::Mesh> mesh
        = create_mesh(MPI_COMM_WORLD, ndofs, strong_scaling, 1);
    t0.stop();

    // Create Poisson problem
    auto data = poisson::problem(mesh);
    A = std::make_shared<dolfin::la::PETScMatrix>(std::move(std::get<0>(data)));
    b = std::make_shared<dolfin::la::PETScVector>(std::move(std::get<1>(data)));
    u = std::get<2>(data);
  }
  else if (problem_type == "elasticity")
  {
    dolfin::common::Timer t0("ZZZ Create Mesh");
    std::shared_ptr<dolfin::mesh::Mesh> mesh
        = create_mesh(MPI_COMM_WORLD, ndofs, strong_scaling, 3);
    t0.stop();

    // Create elasticity problem. Near-nullspace will be attached to the
    // linear operator (matrix).
    auto data = elastic::problem(mesh);
    A = std::make_shared<dolfin::la::PETScMatrix>(std::move(std::get<0>(data)));
    b = std::make_shared<dolfin::la::PETScVector>(std::move(std::get<1>(data)));
    u = std::get<2>(data);
  }
  else
    throw std::runtime_error("Unknown problem type: " + problem_type);

  // Print simulation summary
  if (dolfin::MPI::rank(MPI_COMM_WORLD) == 0)
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
              << u->function_space()->dim() / dolfin::MPI::size(MPI_COMM_WORLD)
              << std::endl;
    std::cout
        << "----------------------------------------------------------------"
        << std::endl;
  }

  // Create solver
  dolfin::la::PETScKrylovSolver solver(MPI_COMM_WORLD);
  solver.set_from_options();
  solver.set_operator(A->mat());

  // Solve
  dolfin::common::Timer t5("ZZZ Solve");
  std::size_t num_iter = solver.solve(u->vector().vec(), b->vec());
  t5.stop();

  if (output)
  {
    dolfin::common::Timer t6("ZZZ Output");
    //  Save solution in XDMF format
    std::string filename
        = output_dir + "/solution-" + std::to_string(num_processes) + ".xdmf";
    dolfin::io::XDMFFile file(MPI_COMM_WORLD, filename);
    file.write(*u);
    t6.stop();
  }

  // Display timings
  dolfin::list_timings({dolfin::TimingType::wall});

  // Report number of Krylov iterations
  if (dolfin::MPI::rank(MPI_COMM_WORLD) == 0)
    std::cout << "*** Number of Krylov iterations: " << num_iter << std::endl;

  return 0;
}

// Copyright(C) 2023-2025 Igor A. Baratta, Chris N. Richardson, Joseph P. Dean,
// Garth N. Wells
// SPDX-License-Identifier:    MIT

#include <thrust/sequence.h>

#include "poisson.h"

#include "src/csr.hpp"
#include "src/laplacian.hpp"
#include "src/mesh.hpp"
#include "src/util.hpp"
#include "src/vector.hpp"

#include <array>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <dolfinx.h>
#include <dolfinx/fem/dolfinx_fem.h>
#include <dolfinx/io/XDMFFile.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/generation.h>
#include <iostream>
#include <memory>
#include <mpi.h>
#include <random>

using namespace dolfinx;
using T = SCALAR_TYPE;
namespace po = boost::program_options;

int main(int argc, char* argv[])
{
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "print usage message")(
      "ndofs", po::value<std::size_t>()->default_value(343),
      "number of dofs per rank")("qmode",
                                 po::value<std::size_t>()->default_value(0),
                                 "quadrature mode (0=p+1, 1=p+2)")(
      "nreps", po::value<std::size_t>()->default_value(1000),
      "number of repetitions")("order",
                               po::value<std::size_t>()->default_value(3),
                               "Polynomial degree (Q)")(
      "mat_comp", po::bool_switch()->default_value(false),
      "Compare result to matrix operator")(
      "batch_size", po::value<std::size_t>()->default_value(0),
      "The geometry batch size. Set to 0 to precompute")(
      "geom_perturb_fact", po::value<T>()->default_value(0.0),
      "Factor of cell diameter to perturb the geometry by")(
      "use_gauss", po::bool_switch()->default_value(false),
      "Use Gauss quadrature");

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv)
                .options(desc)
                .allow_unregistered()
                .run(),
            vm);
  po::notify(vm);

  if (vm.count("help"))
  {
    std::cout << desc << std::endl;
    return 0;
  }

  const std::size_t ndofs = vm["ndofs"].as<std::size_t>();
  const std::size_t nreps = vm["nreps"].as<std::size_t>();
  const std::size_t batch_size = vm["batch_size"].as<std::size_t>();
  const std::size_t order = vm["order"].as<std::size_t>();
  const bool matrix_comparison = vm["mat_comp"].as<bool>();
  const T geom_perturb_fact = vm["geom_perturb_fact"].as<T>();
  const bool use_gauss = vm["use_gauss"].as<bool>();

  // Quadrature mode (qmode=0: nq = P + 1, qmode=1: nq = P + 2)
  const std::size_t qmode = vm["qmode"].as<std::size_t>();
  if (qmode > 1)
    throw std::runtime_error("Invalid qmode");

  init_logging(argc, argv);
  MPI_Init(&argc, &argv);
  {
    MPI_Comm comm{MPI_COMM_WORLD};
    int rank = 0, size = 0;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    double nx_approx = (std::pow(ndofs * size, 1.0 / 3.0) - 1) / order;
    std::int64_t n0 = static_cast<std::int64_t>(nx_approx);
    std::array<std::int64_t, 3> nx = {n0, n0, n0};

    // Try to improve fit to ndofs +/- 5 in each direction
    if (n0 > 5)
    {
      std::int64_t best_misfit
          = (n0 * order + 1) * (n0 * order + 1) * (n0 * order + 1)
            - ndofs * size;
      best_misfit = std::abs(best_misfit);
      for (std::int64_t nx0 = n0 - 5; nx0 < n0 + 6; ++nx0)
      {
        for (std::int64_t ny0 = n0 - 5; ny0 < n0 + 6; ++ny0)
        {
          for (std::int64_t nz0 = n0 - 5; nz0 < n0 + 6; ++nz0)
          {
            std::int64_t misfit
                = (nx0 * order + 1) * (ny0 * order + 1) * (nz0 * order + 1)
                  - ndofs * size;
            if (std::abs(misfit) < best_misfit)
            {
              best_misfit = std::abs(misfit);
              nx = {nx0, ny0, nz0};
            }
          }
        }
      }
    }

    spdlog::info("Mesh shape: {}x{}x{}", nx[0], nx[1], nx[2]);

    // Finite element for higher-order discretisation
    auto element = basix::create_tp_element<T>(
        basix::element::family::P, basix::cell::type::hexahedron, order,
        basix::element::lagrange_variant::gll_warped,
        basix::element::dpc_variant::unset, false);

    // First order coordinate element
    auto element_1
        = std::make_shared<basix::FiniteElement<T>>(basix::create_tp_element<T>(
            basix::element::family::P, basix::cell::type::hexahedron, 1,
            basix::element::lagrange_variant::gll_warped,
            basix::element::dpc_variant::unset, false));
    dolfinx::fem::CoordinateElement<T> coord_element(element_1);

    // Create mesh with overlap region
    std::shared_ptr<mesh::Mesh<T>> mesh;
    {
      mesh::Mesh<T> base_mesh = mesh::create_box<T>(
          comm, {{{0, 0, 0}, {1, 1, 1}}}, {nx[0], nx[1], nx[2]},
          mesh::CellType::hexahedron);

      if (geom_perturb_fact != 0.0)
      {
        const double perturb_x = geom_perturb_fact * 1 / nx[0];
        std::span<T> geom_x = base_mesh.geometry().x();
        std::mt19937 generator(42);
        std::uniform_real_distribution<T> distribution(-perturb_x, perturb_x);
        for (int i = 0; i < geom_x.size(); i += 3)
          geom_x[i] += distribution(generator);
      }

      mesh = std::make_shared<mesh::Mesh<T>>(
          ghost_layer_mesh(base_mesh, coord_element));
    }

    auto V = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace(
        mesh, std::make_shared<const fem::FiniteElement<T>>(element)));
    auto [lcells, bcells] = compute_boundary_cells(V);
    spdlog::debug("lcells = {}, bcells = {}", lcells.size(), bcells.size());

    auto topology = V->mesh()->topology_mutable();
    int tdim = topology->dim();
    std::size_t ncells = mesh->topology()->index_map(tdim)->size_global();
    std::size_t ndofs_global = V->dofmap()->index_map->size_global();

    std::string fp_type = "float";
    if (std::is_same_v<T, float>)
      fp_type += "32";
    else if (std::is_same_v<T, double>)
      fp_type += "64";

    if (rank == 0)
    {
      std::cout << device_information();
      std::cout << "-----------------------------------\n";
      std::cout << "Polynomial degree : " << order << "\n";
#ifndef USE_SLICED
      std::cout << "Sliced : no" << std::endl;
#else
      std::cout << "Sliced : yes" << std::endl;
      std::cout << "Slice size: " << SLICE_SIZE << std::endl;
#endif
      std::cout << "Number of ranks : " << size << "\n";
      std::cout << "Number of cells-global : " << ncells << "\n";
      std::cout << "Number of dofs-global : " << ndofs_global << "\n";
      std::cout << "Number of cells-rank : " << ncells / size << "\n";
      std::cout << "Number of dofs-rank : " << ndofs_global / size << "\n";
      std::cout << "Number of repetitions : " << nreps << "\n";
      std::cout << "Geometry batch size: " << batch_size << "\n";
      std::cout << "Scalar Type: " << fp_type << "\n";
      std::cout << "-----------------------------------\n";
      std::cout << std::flush;
    }

    // Prepare and set Constants for the bilinear form
    auto kappa = std::make_shared<fem::Constant<T>>(2.0);
    auto f = std::make_shared<fem::Function<T>>(V);

    spdlog::debug("Define forms");
    // Define variational forms

    std::vector<ufcx_form*> aforms;
    std::vector<ufcx_form*> Lforms;
    basix::quadrature::type quad_type;
    if (use_gauss)
    {
      quad_type = basix::quadrature::type::gauss_jacobi;
      if (qmode == 0)
      {
        aforms = {form_poisson_a_1_2_GL, form_poisson_a_2_3_GL,
                  form_poisson_a_3_4_GL, form_poisson_a_4_5_GL,
                  form_poisson_a_5_6_GL, form_poisson_a_6_7_GL,
                  form_poisson_a_7_8_GL};
        Lforms = {form_poisson_L_1_2_GL, form_poisson_L_2_3_GL,
                  form_poisson_L_3_4_GL, form_poisson_L_4_5_GL,
                  form_poisson_L_5_6_GL, form_poisson_L_6_7_GL,
                  form_poisson_L_7_8_GL};
      }
      else
      {
        aforms = {form_poisson_a_1_3_GL, form_poisson_a_2_4_GL,
                  form_poisson_a_3_5_GL, form_poisson_a_4_6_GL,
                  form_poisson_a_5_7_GL, form_poisson_a_6_8_GL,
                  form_poisson_a_7_9_GL};
        Lforms = {form_poisson_L_1_3_GL, form_poisson_L_2_4_GL,
                  form_poisson_L_3_5_GL, form_poisson_L_4_6_GL,
                  form_poisson_L_5_7_GL, form_poisson_L_6_8_GL,
                  form_poisson_L_7_9_GL};
      }
    }
    else
    {
      quad_type = basix::quadrature::type::gll;
      if (qmode == 0)
      {
        aforms = {form_poisson_a_1_2_GLL, form_poisson_a_2_3_GLL,
                  form_poisson_a_3_4_GLL, form_poisson_a_4_5_GLL,
                  form_poisson_a_5_6_GLL, form_poisson_a_6_7_GLL,
                  form_poisson_a_7_8_GLL};
        Lforms = {form_poisson_L_1_2_GLL, form_poisson_L_2_3_GLL,
                  form_poisson_L_3_4_GLL, form_poisson_L_4_5_GLL,
                  form_poisson_L_5_6_GLL, form_poisson_L_6_7_GLL,
                  form_poisson_L_7_8_GLL};
      }
      else
      {
        aforms = {form_poisson_a_1_3_GLL, form_poisson_a_2_4_GLL,
                  form_poisson_a_3_5_GLL, form_poisson_a_4_6_GLL,
                  form_poisson_a_5_7_GLL, form_poisson_a_6_8_GLL,
                  form_poisson_a_7_9_GLL};
        Lforms = {form_poisson_L_1_3_GLL, form_poisson_L_2_4_GLL,
                  form_poisson_L_3_5_GLL, form_poisson_L_4_6_GLL,
                  form_poisson_L_5_7_GLL, form_poisson_L_6_8_GLL,
                  form_poisson_L_7_9_GLL};
      }
    }

    auto a = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *aforms.at(order - 1), {V, V}, {}, {{"c0", kappa}}, {}, {}));
    auto L = std::make_shared<fem::Form<T>>(fem::create_form<T>(
        *Lforms.at(order - 1), {V}, {{"w0", f}}, {}, {}, {}));

    spdlog::debug("Interpolate (rank {})", rank);
    f->interpolate(
        [](auto x) -> std::pair<std::vector<T>, std::vector<std::size_t>>
        {
          std::vector<T> out;
          for (std::size_t p = 0; p < x.extent(1); ++p)
          {
            auto dx = (x(0, p) - 0.5) * (x(0, p) - 0.5);
            auto dy = (x(1, p) - 0.5) * (x(1, p) - 0.5);
            out.push_back(1000 * std::exp(-(dx + dy) / 0.02));
          }
          return {out, {out.size()}};
        });

    int fdim = tdim - 1;
    spdlog::debug("Create f->c on {}", rank);
    topology->create_connectivity(fdim, tdim);
    spdlog::debug("Done f->c on {}", rank);

    auto dofmap = V->dofmap();
    auto facets = dolfinx::mesh::exterior_facet_indices(*topology);
    auto bdofs = fem::locate_dofs_topological(*topology, *dofmap, fdim, facets);
    auto bc = std::make_shared<const fem::DirichletBC<T>>(1.3, bdofs, V);

    // Copy data to GPU
    // Constants
    // TODO Pack these properly
    const int num_cells_all = mesh->topology()->index_map(tdim)->size_local()
                              + mesh->topology()->index_map(tdim)->num_ghosts();
    thrust::device_vector<T> constants_d(num_cells_all, kappa->value[0]);
    std::span<const T> constants_d_span(
        thrust::raw_pointer_cast(constants_d.data()), constants_d.size());
    spdlog::info("Send constants to GPU (size = {} bytes)",
                 constants_d.size() * sizeof(T));

    // V dofmap
    thrust::device_vector<std::int32_t> dofmap_d(dofmap->map().data_handle(),
                                                 dofmap->map().data_handle()
                                                     + dofmap->map().size());
    std::span<const std::int32_t> dofmap_d_span(
        thrust::raw_pointer_cast(dofmap_d.data()), dofmap_d.size());
    spdlog::info("Send dofmap to GPU (size = {} bytes)",
                 dofmap_d.size() * sizeof(std::int32_t));

    // Define vectors
    using DeviceVector = dolfinx::acc::Vector<T, acc::Device::HIP>;

    // Input vector
    auto map = V->dofmap()->index_map;
    spdlog::info("Create device vector u");

    DeviceVector u(map, 1);
    u.set(T{0.0});

    // Output vector
    spdlog::info("Create device vector y");
    DeviceVector y(map, 1);
    y.set(T{0.0});

    // -----------------------------------------------------------------------------
    // Put geometry information onto device
    const fem::CoordinateElement<T>& cmap = mesh->geometry().cmap();
    auto xdofmap = mesh->geometry().dofmap();

    // Geometry dofmap
    spdlog::info("Copy geometry dofmap to device ({} bytes)",
                 xdofmap.size() * sizeof(std::int32_t));
    thrust::device_vector<std::int32_t> xdofmap_d(
        xdofmap.data_handle(), xdofmap.data_handle() + xdofmap.size());
    std::span<const std::int32_t> xdofmap_d_span(
        thrust::raw_pointer_cast(xdofmap_d.data()), xdofmap_d.size());
    // Geometry points
    spdlog::info("Copy geometry to device ({} bytes)",
                 mesh->geometry().x().size() * sizeof(T));
    thrust::device_vector<T> xgeom_d(mesh->geometry().x().begin(),
                                     mesh->geometry().x().end());
    std::span<const T> xgeom_d_span(thrust::raw_pointer_cast(xgeom_d.data()),
                                    xgeom_d.size());

    // TODO Ghosts
    const int num_dofs = map->size_local() + map->num_ghosts();
    std::vector<std::int8_t> bc_marker(num_dofs, 0);
    bc->mark_dofs(bc_marker);
    thrust::device_vector<std::int8_t> bc_marker_d(bc_marker.begin(),
                                                   bc_marker.end());
    std::span<const std::int8_t> bc_marker_d_span(
        thrust::raw_pointer_cast(bc_marker_d.data()), bc_marker_d.size());

    // Create matrix free operator
    spdlog::info("Create MatFreeLaplacian");
    dolfinx::common::Timer op_create_timer("% Create matfree operator");
    acc::MatFreeLaplacian<T> op(order, qmode, constants_d_span, dofmap_d_span,
                                xgeom_d_span, xdofmap_d_span, cmap, lcells,
                                bcells, bc_marker_d_span, quad_type,
                                batch_size);

    op_create_timer.stop();

    la::Vector<T> b(map, 1);
    fem::assemble_vector(b.mutable_array(), *L);
    //    fem::apply_lifting<T, T>(b.mutable_array(), {a}, {{bc}}, {}, T(1));
    // b.scatter_rev(std::plus<T>());
    u.copy_from_host(b); // Copy data from host vector to device vector
    u.scatter_fwd_begin();

    // Matrix free
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < nreps; ++i)
      op(u, y);
    auto stop = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = stop - start;
    std::cout << "Mat-free Matvec time: " << duration.count() << std::endl;
    std::cout << "Mat-free action Gdofs/s: "
              << ndofs_global * nreps / (1e9 * duration.count()) << std::endl;

    std::cout << "Norm of u = " << acc::norm(u) << std::endl;
    std::cout << "Norm of y = " << acc::norm(y) << std::endl;

    if (matrix_comparison)
    {
      // Compare to assembling on CPU and copying matrix to GPU
      DeviceVector z(map, 1);
      z.set(T{0.0});

      acc::MatrixOperator<T> mat_op(a, {*bc});
      dolfinx::common::Timer mtimer("% CSR Matvec");
      for (int i = 0; i < nreps; ++i)
        mat_op(u, z);
      mtimer.stop();

      std::cout << "Norm of u = " << acc::norm(u) << "\n";
      std::cout << "Norm of z = " << acc::norm(z) << "\n";

      // Compute error
      DeviceVector e(map, 1);
      acc::axpy(e, T{-1.0}, y, z);
      std::cout << "Norm of error = " << acc::norm(e) << "\n";
      std::cout << "Relative norm of error = " << acc::norm(e) / acc::norm(z)
                << "\n";

      // Compute erorr in diagonal computation
      DeviceVector mat_free_inv_diag(map, 1);
      op.get_diag_inverse(mat_free_inv_diag);
      DeviceVector mat_inv_diag(map, 1);
      mat_op.get_diag_inverse(mat_inv_diag);

      DeviceVector e_diag(map, 1);
      acc::axpy(e_diag, T{-1.0}, mat_inv_diag, mat_free_inv_diag);
      std::cout << "Norm of diagonal error = " << acc::norm(e_diag) << "\n";
    }

    // Display timings
    dolfinx::list_timings(MPI_COMM_WORLD);
  }

  MPI_Finalize();

  return 0;
}

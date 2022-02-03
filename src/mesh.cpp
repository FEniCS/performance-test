// Copyright (C) 2019 Chris N. Richardson and Garth N. Wells
// Licensed under the MIT License. See LICENSE file in the project
// root for full license information.

#include "mesh.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/graph/partitioners.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/graphbuild.h>
#include <dolfinx/refinement/refine.h>
#include <memory>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace
{
// Calculate number of vertices, edges, facets, and cells for any given
// level of refinement
constexpr std::tuple<std::int64_t, std::int64_t, std::int64_t, std::int64_t>
num_entities(std::int64_t i, std::int64_t j, std::int64_t k, int nrefine)
{
  std::int64_t nv = (i + 1) * (j + 1) * (k + 1);
  std::int64_t ne = 0;
  std::int64_t nc = (i * j * k) * 6;
  std::int64_t earr[3] = {1, 3, 7};
  std::int64_t farr[2] = {2, 12};
  for (int r = 0; r < nrefine; ++r)
  {
    ne = earr[0] * (i + j + k) + earr[1] * (i * j + j * k + k * i)
         + earr[2] * i * j * k;
    nv += ne;
    nc *= 8;
    earr[0] *= 2;
    earr[1] *= 4;
    earr[2] *= 8;
    farr[0] *= 4;
    farr[1] *= 8;
  }
  ne = earr[0] * (i + j + k) + earr[1] * (i * j + j * k + k * i)
       + earr[2] * i * j * k;
  std::int64_t nf = farr[0] * (i * j + j * k + k * i) + farr[1] * i * j * k;

  return {nv, ne, nf, nc};
}

std::int64_t num_pdofs(std::int64_t i, std::int64_t j, std::int64_t k,
                       int nrefine, int order)
{
  auto [nv, ne, nf, nc] = num_entities(i, j, k, nrefine);

  switch (order)
  {
  case 1:
    return nv;
  case 2:
    return nv + ne;
  case 3:
    return nv + 2 * ne + nf;
  case 4:
    return nv + 3 * ne + 3 * nf + nc;
  default:
    throw std::runtime_error("Order not supported");
  }
}

} // namespace

dolfinx::mesh::Mesh create_cube_mesh(MPI_Comm comm, std::size_t target_dofs,
                                     bool target_dofs_total,
                                     std::size_t dofs_per_node, int order)
{
  // Get number of processes
  const std::size_t num_processes = dolfinx::MPI::size(comm);

  // Target total dofs
  std::int64_t N = 0;
  if (target_dofs_total == true)
    N = target_dofs / dofs_per_node;
  else
    N = target_dofs * num_processes / dofs_per_node;

  std::size_t Nx, Ny, Nz;
  int r = 0;

  // Get initial guess for Nx, Ny, Nz, r
  Nx = 1;
  std::int64_t nc = 0;
  while (nc < N)
  {
    ++Nx;
    if (Nx > 1000)
    {
      Nx = 40;
      ++r;
    }
    nc = num_pdofs(Nx, Nx, Nx, r, order);
  }

  Ny = Nx;
  Nz = Nx;

  std::size_t i0 = Nx - 10;
  std::size_t mindiff = 1000000;
  for (std::size_t i = i0; i < i0 + 20; ++i)
  {
    for (std::size_t j = i - 5; j < i + 5; ++j)
    {
      for (std::size_t k = i - 5; k < i + 5; ++k)
      {
        std::size_t diff = std::abs(num_pdofs(i, j, k, r, order) - N);
        if (diff < mindiff)
        {
          mindiff = diff;
          Nx = i;
          Ny = j;
          Nz = k;
        }
      }
    }
  }

#ifdef HAS_PARMETIS
  auto graph_part = dolfinx::graph::parmetis::partitioner();
#elif HAS_PTSCOTCH
  auto graph_part = dolfinx::graph::scotch::partitioner(
      dolfinx::graph::scotch::strategy::scalability);
#elif HAS_KAHIP
  auto graph_part = dolfinx::graph::kahip::partitioner();
#else
#error "No mesh partitioner has been selected"
#endif

  auto cell_part = dolfinx::mesh::create_cell_partitioner(graph_part);
  auto mesh = dolfinx::mesh::create_box(
      comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {Nx, Ny, Nz},
      dolfinx::mesh::CellType::tetrahedron, dolfinx::mesh::GhostMode::none,
      cell_part);

  if (dolfinx::MPI::rank(mesh.comm()) == 0)
  {
    std::cout << "UnitCube (" << Nx << "x" << Ny << "x" << Nz
              << ") to be refined " << r << " times\n";
  }

  for (int i = 0; i < r; ++i)
  {
    mesh.topology_mutable().create_connectivity(3, 1);
    mesh = dolfinx::refinement::refine(mesh, false);
  }

  return mesh;
}
//-----------------------------------------------------------------------------
dolfinx::mesh::Mesh create_mesh_geometric(MPI_Comm comm,
                                          std::size_t target_dofs,
                                          bool target_dofs_total,
                                          std::size_t dofs_per_node)
{
  double target = target_dofs / dofs_per_node;
  const int mpi_size = dolfinx::MPI::size(comm);
  const int mpi_rank = dolfinx::MPI::rank(comm);
  if (!target_dofs_total)
    target *= mpi_size;

  // number of vertices on one edge
  double N_edge = std::pow(target, 1.0 / 3.0);

  // Get cube root of number of cores
  const double px0 = std::pow(static_cast<double>(mpi_size), 1.0 / 3.0);
  int px = static_cast<int>(px0 + 0.5);
  if (px == 1)
    throw std::runtime_error("Too few cores for geometric partitioning");

  while (mpi_size % px > 0)
  {
    --px;
    if (px == 1)
      throw std::runtime_error(
          "Unsuitable number of cores for geometric partitioning");
  }

  const int pypz = mpi_size / px;
  const double py0 = std::pow(static_cast<double>(pypz), 0.5);
  int py = static_cast<int>(py0 + 0.5);

  while (pypz % py > 0)
  {
    --py;
    if (py == 1)
      throw std::runtime_error(
          "Unsuitable number of cores for geometric partitioning");
  }

  int pz = pypz / py;

  // Local number of cells in each direction
  std::int64_t nx = N_edge / px;
  std::int64_t ny = N_edge / py;
  std::int64_t nz = N_edge / pz;

  // Global size
  std::int64_t NX = nx * px;
  std::int64_t NY = ny * py;
  std::int64_t NZ = nz * pz;

  if (mpi_rank == 0)
  {
    std::cout << "Mesh size = " << NX * NY * NZ << std::endl;
    std::cout << "Mesh size per core = " << NX * NY * NZ / mpi_size
              << std::endl;
  }

  // Position in global space
  int mx = mpi_rank / pypz;
  int my = (mpi_rank % pypz) / pz;
  int mz = (mpi_rank % pypz) % pz;

  dolfinx::common::Timer timer("Build BoxMesh");
  std::vector<std::int64_t> cells(6 * nx * ny * nz * 4);

  // Create tetrahedra
  std::size_t cp = 0;
  for (std::int64_t i = 0; i < nx; ++i)
    for (std::int64_t j = 0; j < ny; ++j)
      for (std::int64_t k = 0; k < nz; ++k)
      {
        const int ix = mx * nx + i;
        const int iy = my * ny + j;
        const int iz = mz * nz + k;

        const std::int64_t v0 = iz * (NX + 1) * (NY + 1) + iy * (NX + 1) + ix;
        const std::int64_t v1 = v0 + 1;
        const std::int64_t v2 = v0 + (NX + 1);
        const std::int64_t v3 = v1 + (NX + 1);
        const std::int64_t v4 = v0 + (NX + 1) * (NY + 1);
        const std::int64_t v5 = v1 + (NX + 1) * (NY + 1);
        const std::int64_t v6 = v2 + (NX + 1) * (NY + 1);
        const std::int64_t v7 = v3 + (NX + 1) * (NY + 1);

        // Note that v0 < v1 < v2 < v3 < vmid
        xt::xtensor_fixed<std::int64_t, xt::xshape<6, 4>> c
            = {{v0, v1, v3, v7}, {v0, v1, v7, v5}, {v0, v5, v7, v4},
               {v0, v3, v2, v7}, {v0, v6, v4, v7}, {v0, v2, v6, v7}};

        std::copy(c.begin(), c.end(), std::next(cells.begin(), cp));
        cp += 24;
      }

  std::vector<std::int32_t> offsets(6 * nx * ny * nz + 1, 0);
  for (std::size_t i = 0; i < offsets.size() - 1; ++i)
    offsets[i + 1] = offsets[i] + 4;

  dolfinx::fem::CoordinateElement element(dolfinx::mesh::CellType::tetrahedron,
                                          1);

  auto partitioner =
      [](MPI_Comm comm, int, int tdim,
         const dolfinx::graph::AdjacencyList<std::int64_t>& cell_topology,
         dolfinx::mesh::GhostMode) {
        // Find out the ghosting information
        auto [graph, _]
            = dolfinx::mesh::build_dual_graph(comm, cell_topology, tdim);

        // FIXME: much of this is reverse engineering of data that is already
        // known in the GraphBuilder

        const int mpi_size = dolfinx::MPI::size(comm);
        const int mpi_rank = dolfinx::MPI::rank(comm);
        const std::int32_t local_size = graph.num_nodes();
        std::vector<std::int32_t> local_sizes(mpi_size);
        std::vector<std::int64_t> local_offsets(mpi_size + 1);

        // Get the "local range" for all processes
        MPI_Allgather(&local_size, 1, MPI_INT32_T, local_sizes.data(), 1,
                      MPI_INT32_T, comm);
        for (int i = 0; i < mpi_size; ++i)
          local_offsets[i + 1] = local_offsets[i] + local_sizes[i];

        // All cells should go to their currently assigned ranks (no change)
        // but must also be sent to their ghost destinations, which are
        // determined here.
        std::vector<std::int32_t> destinations;
        destinations.reserve(graph.num_nodes());
        std::vector<std::int32_t> dest_offsets = {0};
        dest_offsets.reserve(graph.num_nodes());
        for (int i = 0; i < graph.num_nodes(); ++i)
        {
          destinations.push_back(mpi_rank);
          for (int j = 0; j < graph.num_links(i); ++j)
          {
            std::int64_t index = graph.links(i)[j];
            if (index < local_offsets[mpi_rank]
                or index >= local_offsets[mpi_rank + 1])
            {
              // Ghosted cell - identify which process it should be sent to.
              for (std::size_t k = 0; k < local_offsets.size(); ++k)
              {
                if (index >= local_offsets[k] and index < local_offsets[k + 1])
                {
                  destinations.push_back(k);
                  break;
                }
              }
            }
          }
          dest_offsets.push_back(destinations.size());
        }

        return dolfinx::graph::AdjacencyList<std::int32_t>(
            std::move(destinations), std::move(dest_offsets));
      };

  std::int64_t npoints_global = (NX + 1) * (NY + 1) * (NZ + 1);
  std::array<std::int64_t, 2> range
      = dolfinx::MPI::local_range(mpi_rank, npoints_global, mpi_size);
  int npoints = range[1] - range[0];

  xt::xtensor<double, 2> geom = xt::zeros<double>({npoints, 3});
  std::int64_t nlayer = (NX + 1) * (NY + 1);
  for (std::size_t j = range[0]; j < range[1]; ++j)
  {
    std::int64_t ix = j / nlayer;
    std::int64_t iy = (j % nlayer) / (NY + 1);
    std::int64_t iz = (j % nlayer) % (NY + 1);
    geom(j - range[0], 0)
        = static_cast<double>(ix) / static_cast<double>(NX + 1);
    geom(j - range[0], 1)
        = static_cast<double>(iy) / static_cast<double>(NY + 1);
    geom(j - range[0], 2)
        = static_cast<double>(iz) / static_cast<double>(NZ + 1);
  }

  dolfinx::mesh::Mesh mesh = dolfinx::mesh::create_mesh(
      comm,
      dolfinx::graph::AdjacencyList<std::int64_t>(std::move(cells),
                                                  std::move(offsets)),
      element, geom, dolfinx::mesh::GhostMode::none, partitioner);

  return mesh;
}
//-----------------------------------------------------------------------------
std::shared_ptr<dolfinx::mesh::Mesh>
create_spoke_mesh(MPI_Comm comm, std::size_t target_dofs,
                  bool target_dofs_total, std::size_t dofs_per_node)
{
  int target = target_dofs / dofs_per_node;
  int mpi_size = dolfinx::MPI::size(comm);
  if (!target_dofs_total)
    target *= mpi_size;

  // Parameters controlling shape
  constexpr int n = 17;       // number of spokes
  constexpr double r0 = 0.25; // inner radius of ring
  constexpr double r1 = 0.5;  // outer radius of ring

  constexpr double h0 = 1.2; // height (inner)
  constexpr double h1 = 1.0; // height (outer)

  constexpr int lspur = 6;     // number of elements in each spoke
  constexpr double l0 = 0.5;   // length of each element in spoke
  constexpr double dth = 0.15; // curl (angle increment) as spoke goes out
  constexpr double tap
      = 0.9; // taper (fractional height decrease on each element)

  // Subdivision of a cube into 6 tetrahedra
  constexpr int cube[6][4] = {{0, 1, 2, 4}, {1, 2, 4, 5}, {2, 4, 5, 6},
                              {0, 2, 3, 4}, {6, 7, 4, 2}, {2, 3, 4, 7}};

  // Calculate number of points and cells (only on process 0)
  int npoints = 0;
  int ncells = 0;
  const int mpi_rank = dolfinx::MPI::rank(comm);

  if (mpi_rank == 0)
  {
    npoints = n * 4 + n * lspur * 4;
    ncells = n * 6 + n * lspur * 6;
  }

  xt::xtensor<double, 2> geom = xt::zeros<double>({npoints, 3});
  std::vector<std::int64_t> topo(4 * ncells);
  if (mpi_rank == 0)
  {
    int p = 0;
    int c = 0;

    // Add n 'cubes' to make a joined up ring.
    for (int i = 0; i < n; ++i)
    {
      std::cout << "Adding cube " << i << "\n";
      // Get the points for current cube
      std::array<int, 8> pts;
      for (int j = 0; j < pts.size(); ++j)
        pts[j] = ((i * 4 + j) % (n * 4));

      // Add to topology
      for (int k = 0; k < 6; ++k)
      {
        for (int j = 0; j < 4; ++j)
          topo[4 * c + j] = pts[cube[k][j]];
        ++c;
      }

      // Calculate the position of points
      double th = 2 * M_PI * i / n;
      xt::xtensor_fixed<double, xt::xshape<3>> point
          = {r0 * cos(th), r0 * sin(th), h0};

      xt::row(geom, p++) = point;
      point = {r0 * cos(th), r0 * sin(th), -h0};
      xt::row(geom, p++) = point;
      point = {r1 * cos(th), r1 * sin(th), -h1};
      xt::row(geom, p++) = point;
      point = {r1 * cos(th), r1 * sin(th), h1};
      xt::row(geom, p++) = point;
    }

    // Add spurs to ring
    for (int i = 0; i < n; ++i)
    {
      std::cout << "Adding spur " << i << "\n";

      // Intermediate angle between two faces
      double th0 = 2 * M_PI * (i + .5) / n;

      // Starting points on outer edge of ring
      xt::xtensor_fixed<int, xt::xshape<8>> pts = {(i * 4 + 2) % (n * 4),
                                                   (i * 4 + 3) % (n * 4),
                                                   (i * 4 + 7) % (n * 4),
                                                   (i * 4 + 6) % (n * 4),
                                                   0,
                                                   0,
                                                   0,
                                                   0};

      // Build each spur outwards
      for (int k = 0; k < lspur; ++k)
      {
        // Add new points
        for (int j = 0; j < 4; ++j)
        {
          pts[j + 4] = p;
          xt::row(geom, p) = xt::row(geom, pts[j]);
          geom(p, 0) += l0 * cos(th0 + k * dth);
          geom(p, 1) += l0 * sin(th0 + k * dth);
          geom(p, 2) *= pow(tap, k);
          ++p;
        }

        // Add new cells
        for (int m = 0; m < 6; ++m)
        {
          for (int j = 0; j < 4; ++j)
            topo[4 * c + j] = pts[cube[m][j]];
          ++c;
        }

        // Outer face becomes inner face of next cube
        using namespace xt::placeholders;
        xt::view(pts, xt::range(0, 4)) = xt::view(pts, xt::range(-4, _));
      }
    }

    // Check geometric sizes and rescale
    xt::col(geom, 0) -= 0.9 * xt::amin(xt::col(geom, 0));
    double scaling = 0.9 * xt::amax(xt::col(geom, 0))[0];
    geom /= scaling;

    LOG(INFO) << "x range = " << xt::amin(xt::col(geom, 0))() << " - "
              << xt::amax(xt::col(geom, 0))() << "\n";
    LOG(INFO) << "y range = " << xt::amin(xt::col(geom, 1))() << " - "
              << xt::amax(xt::col(geom, 1))() << "\n";
    LOG(INFO) << "z range = " << xt::amin(xt::col(geom, 2))() << " - "
              << xt::amax(xt::col(geom, 2))() << "\n";
  }

  // New Mesh
  std::vector<std::int32_t> offsets(ncells + 1, 0);
  for (std::size_t i = 0; i < offsets.size() - 1; ++i)
    offsets[i + 1] = offsets[i] + 4;

  dolfinx::fem::CoordinateElement element(dolfinx::mesh::CellType::tetrahedron,
                                          1);

  auto mesh = std::make_shared<dolfinx::mesh::Mesh>(dolfinx::mesh::create_mesh(
      comm,
      dolfinx::graph::AdjacencyList<std::int64_t>(std::move(topo),
                                                  std::move(offsets)),
      element, geom, dolfinx::mesh::GhostMode::none));

  mesh->topology_mutable().create_entities(1);

  while (mesh->topology().index_map(0)->size_global()
             + mesh->topology().index_map(1)->size_global()
         < target)
  {
    mesh = std::make_shared<dolfinx::mesh::Mesh>(
        dolfinx::refinement::refine(*mesh, false));
    mesh->topology_mutable().create_entities(1);
  }

  double fraction
      = (double)(target - mesh->topology().index_map(0)->size_global())
        / mesh->topology().index_map(1)->size_global();

  if (mpi_rank == 0)
  {
    std::cout << "Create unstructured mesh: desired fraction=" << fraction
              << std::endl;
  }

  // Estimate step needed to get desired refinement fraction
  // using some heuristics and bisection method
  int nmarked = pow(fraction, 1.6) * 2000;

  double f_lower = 0.0;
  double f_upper = 1.0;
  int lmark = 0;
  int umark = 2000;

  std::shared_ptr<dolfinx::mesh::Mesh> meshi;
  for (int k = 0; k < 5; ++k)
  {
    // Trial step
    mesh->topology_mutable().create_entities(1);
    std::vector<std::int32_t> marked_edges;
    const std::int32_t num_edges = mesh->topology().index_map(1)->size_local();
    for (int i = 0; i < num_edges; ++i)
      if (i % 2000 < nmarked)
        marked_edges.push_back(i);

    meshi = std::make_shared<dolfinx::mesh::Mesh>(
        dolfinx::refinement::refine(*mesh, marked_edges, false));

    double actual_fraction
        = (double)(meshi->topology().index_map(0)->size_global()
                   - mesh->topology().index_map(0)->size_global())
          / mesh->topology().index_map(1)->size_global();

    if (mpi_rank == 0)
    {
      std::cout << "Edges marked = " << nmarked << "/2000\n";
      std::cout << "Step " << k
                << " achieved actual fraction = " << actual_fraction << "\n";
    }

    if (actual_fraction > fraction)
    {
      umark = nmarked;
      f_upper = actual_fraction;
    }
    else
    {
      lmark = nmarked;
      f_lower = actual_fraction;
    }
    int new_mark = (lmark * (f_upper - fraction) + umark * (fraction - f_lower))
                   / (f_upper - f_lower);

    if (nmarked == new_mark)
      break;
    else
      nmarked = new_mark;
  }

  return meshi;
}

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
#include <dolfinx/refinement/refine.h>
#include <memory>
#include <numbers>
#include <span>

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

dolfinx::mesh::Mesh<double>
create_cube_mesh(MPI_Comm comm, std::size_t target_dofs, bool target_dofs_total,
                 std::size_t dofs_per_node, int order, bool use_subcomm)
{
  // Get number of processes
  const std::size_t num_processes = dolfinx::MPI::size(comm);

  // Target total dofs
  std::int64_t N = 0;
  if (target_dofs_total == true)
    N = target_dofs / dofs_per_node;
  else
    N = target_dofs * num_processes / dofs_per_node;

  std::int64_t Nx, Ny, Nz;
  int r = 0;

  // Choose Nx_max carefully. If too large, the base mesh may become too
  // large for the partitioner; likewise, if too small, it will fail on
  // large numbers of processes.
  const std::int64_t Nx_max = 200;

  // Get initial guess for Nx, Ny, Nz, r
  Nx = 1;
  std::int64_t ndofs = 0;
  while (ndofs < N)
  {
    // Increase base mesh size
    ++Nx;
    if (Nx > Nx_max)
    {
      // Base mesh got too big, so add refinement levels
      // Each increase will dramatically (~8x) increase the number of
      // dofs
      while (ndofs < N)
      {
        // Keep on refining until we have overshot
        ++r;
        ndofs = num_pdofs(Nx, Nx, Nx, r, order);
      }
      while (ndofs > N)
      {
        // Shrink base mesh until dofs are back on target
        --Nx;
        ndofs = num_pdofs(Nx, Nx, Nx, r, order);
      }
    }
    ndofs = num_pdofs(Nx, Nx, Nx, r, order);
  }

  Ny = Nx;
  Nz = Nx;

  // Optimise number of dofs by trying nearby mesh sizes +/- 5 or 10 in
  // each dimension

  std::size_t mindiff = 1000000;
  for (std::int64_t i = Nx - 10; i < Nx + 10; ++i)
  {
    for (std::int64_t j = i - 5; j < i + 5; ++j)
    {
      for (std::int64_t k = i - 5; k < i + 5; ++k)
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

  MPI_Comm sub_comm;

  if (use_subcomm)
  {
    // Create a sub-communicator for mesh partitioning
    MPI_Comm shm_comm;
    // Get a local comm on each node
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                        &shm_comm);
    int shm_comm_rank = dolfinx::MPI::rank(shm_comm);
    MPI_Comm_free(&shm_comm);
    // Create a comm across nodes, using rank 0 of the local comm on each node
    int color = (shm_comm_rank == 0) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(comm, color, 0, &sub_comm);
  }
  else
    MPI_Comm_dup(comm, &sub_comm);

  auto cell_part = dolfinx::mesh::create_cell_partitioner(
      dolfinx::mesh::GhostMode::none, graph_part);
  auto mesh = dolfinx::mesh::create_box(
      comm, sub_comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {Nx, Ny, Nz},
      dolfinx::mesh::CellType::tetrahedron, cell_part);

  MPI_Comm_free(&sub_comm);

  if (dolfinx::MPI::rank(mesh.comm()) == 0)
  {
    std::cout << "UnitCube (" << Nx << "x" << Ny << "x" << Nz
              << ") to be refined " << r << " times" << std::endl;
  }

  for (int i = 0; i < r; ++i)
  {
    mesh.topology_mutable()->create_connectivity(3, 1);
    auto [new_mesh, _x, _y] = dolfinx::refinement::refine(mesh, std::nullopt, false);
    mesh = std::move(new_mesh);
  }

  return mesh;
}
//-----------------------------------------------------------------------------
std::shared_ptr<dolfinx::mesh::Mesh<double>>
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

  std::vector<double> x(npoints * 3);
  std::vector<std::int64_t> topo(4 * ncells);
  if (mpi_rank == 0)
  {
    int p = 0;
    int c = 0;

    // Add n 'cubes' to make a joined up ring.
    for (int i = 0; i < n; ++i)
    {
      std::cout << "Adding cube " << i << std::endl;
      // Get the points for current cube
      std::array<int, 8> pts;
      for (std::size_t j = 0; j < pts.size(); ++j)
        pts[j] = (i * 4 + j) % (n * 4);

      // Add to topology
      for (int k = 0; k < 6; ++k)
      {
        for (int j = 0; j < 4; ++j)
          topo[4 * c + j] = pts[cube[k][j]];
        ++c;
      }

      // Calculate the position of points
      const double th = 2 * std::numbers::pi * i / n;

      std::array p0 = {r0 * std::cos(th), r0 * std::sin(th), h0};
      std::copy(p0.begin(), p0.end(), std::next(x.begin(), 3 * p));

      std::array p1 = {r0 * std::cos(th), r0 * std::sin(th), -h0};
      std::copy(p1.begin(), p1.end(), std::next(x.begin(), 3 * (p + 1)));

      std::array p2 = {r1 * std::cos(th), r1 * std::sin(th), -h1};
      std::copy(p2.begin(), p2.end(), std::next(x.begin(), 3 * (p + 2)));

      std::array p3 = {r1 * std::cos(th), r1 * std::sin(th), h1};
      std::copy(p3.begin(), p3.end(), std::next(x.begin(), 3 * (p + 3)));

      p += 4;
    }

    // Add spurs to ring
    for (int i = 0; i < n; ++i)
    {
      std::cout << "Adding spur " << i << std::endl;

      // Intermediate angle between two faces
      const double th0 = 2 * std::numbers::pi * (i + 0.5) / n;

      // Starting points on outer edge of ring
      std::array<int, 8> pts = {(i * 4 + 2) % (n * 4),
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
          std::span<double, 3> xp(x.data() + 3 * p, 3);
          std::copy_n(std::next(x.begin(), 3 * pts[j]), 3, xp.begin());
          xp[0] += l0 * std::cos(th0 + k * dth);
          xp[1] += l0 * std::sin(th0 + k * dth);
          xp[2] *= std::pow(tap, k);
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
        std::span<int, 8> _pts(pts.data(), 8);
        auto pts0 = _pts.first<4>();
        auto pts1 = _pts.last<4>();
        std::copy(pts1.begin(), pts1.end(), pts0.begin());
      }
    }

    // Check geometric sizes and rescale
    double x0min(0), x0max(0), x1min(0), x1max(0), x2min(0), x2max(0);
    for (std::size_t i = 0; i < x.size(); i += 3)
    {
      x0min = std::min(std::abs(x[i]), x0min);
      x0max = std::max(std::abs(x[i]), x0max);

      x1min = std::min(std::abs(x[i + 1]), x1min);
      x1max = std::max(std::abs(x[i + 1]), x1max);

      x2min = std::min(std::abs(x[i + 2]), x2min);
      x2max = std::max(std::abs(x[i + 2]), x2max);
    }

    for (std::size_t i = 0; i < x.size(); i += 3)
      x[i] -= 0.9 * x0min;
    std::transform(x.begin(), x.end(), x.begin(),
                   [scale = 0.9 * x0max](auto x) { return x / scale; });

    spdlog::info("x range = {} - {}", x0min, x0max);
    spdlog::info("y range = {} - {}", x1min, x1max);
    spdlog::info("z range = {} - {}", x2min, x2max);
  }

  // New Mesh
  dolfinx::fem::CoordinateElement<double> element(
      dolfinx::mesh::CellType::tetrahedron, 1);

  auto mesh = std::make_shared<dolfinx::mesh::Mesh<double>>(
      dolfinx::mesh::create_mesh(comm, topo, element, x, {x.size() / 3, 3},
                                 dolfinx::mesh::GhostMode::none));

  mesh->topology_mutable()->create_entities(1);

  while (mesh->topology()->index_map(0)->size_global()
             + mesh->topology()->index_map(1)->size_global()
         < target)
  {
    auto [new_mesh, _x, _y] = dolfinx::refinement::refine(*mesh, std::nullopt, false);
    mesh = std::make_shared<dolfinx::mesh::Mesh<double>>(new_mesh);
    mesh->topology_mutable()->create_entities(1);
  }

  double fraction
      = (double)(target - mesh->topology()->index_map(0)->size_global())
        / mesh->topology()->index_map(1)->size_global();

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

  std::shared_ptr<dolfinx::mesh::Mesh<double>> meshi;
  for (int k = 0; k < 5; ++k)
  {
    // Trial step
    mesh->topology_mutable()->create_entities(1);
    std::vector<std::int32_t> marked_edges;
    const std::int32_t num_edges = mesh->topology()->index_map(1)->size_local();
    for (int i = 0; i < num_edges; ++i)
      if (i % 2000 < nmarked)
        marked_edges.push_back(i);

    auto [new_mesh, _x, _y] = dolfinx::refinement::refine(*mesh, marked_edges, false);
    meshi = std::make_shared<dolfinx::mesh::Mesh<double>>(new_mesh);

    double actual_fraction
        = (double)(meshi->topology()->index_map(0)->size_global()
                   - mesh->topology()->index_map(0)->size_global())
          / mesh->topology()->index_map(1)->size_global();

    if (mpi_rank == 0)
    {
      std::cout << "Edges marked = " << nmarked << "/2000" << std::endl;
      std::cout << "Step " << k
                << " achieved actual fraction = " << actual_fraction
                << std::endl;
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

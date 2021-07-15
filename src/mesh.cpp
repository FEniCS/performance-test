// Copyright (C) 2019 Chris N. Richardson and Garth N. Wells
// Licensed under the MIT License. See LICENSE file in the project
// root for full license information.

#include "mesh.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/log.h>
#include <dolfinx/common/types.h>
#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/ElementDofLayout.h>
#include <dolfinx/generation/BoxMesh.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/MeshTags.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/refinement/refine.h>
#include <memory>
#include <xtensor/xfixed.hpp>

namespace
{

//-----------------------------------------------------------------------------
xt::xtensor<double, 2>
create_geom(MPI_Comm comm, const std::array<std::array<double, 3>, 2>& p,
            std::array<std::size_t, 3> n)
{
  // Extract data
  const std::array<double, 3>& p0 = p[0];
  const std::array<double, 3>& p1 = p[1];
  std::int64_t nx = n[0];
  std::int64_t ny = n[1];
  std::int64_t nz = n[2];

  const std::int64_t n_points = (nx + 1) * (ny + 1) * (nz + 1);
  std::array range_p = dolfinx::MPI::local_range(
      dolfinx::MPI::rank(comm), n_points, dolfinx::MPI::size(comm));

  // Extract minimum and maximum coordinates
  const double x0 = std::min(p0[0], p1[0]);
  const double x1 = std::max(p0[0], p1[0]);
  const double y0 = std::min(p0[1], p1[1]);
  const double y1 = std::max(p0[1], p1[1]);
  const double z0 = std::min(p0[2], p1[2]);
  const double z1 = std::max(p0[2], p1[2]);

  const double a = x0;
  const double b = x1;
  const double ab = (b - a) / static_cast<double>(nx);
  const double c = y0;
  const double d = y1;
  const double cd = (d - c) / static_cast<double>(ny);
  const double e = z0;
  const double f = z1;
  const double ef = (f - e) / static_cast<double>(nz);

  xt::xtensor<double, 2> geom(
      {static_cast<std::size_t>(range_p[1] - range_p[0]), 3});
  const std::int64_t sqxy = (nx + 1) * (ny + 1);
  std::array<double, 3> point;
  for (std::int64_t v = range_p[0]; v < range_p[1]; ++v)
  {
    const std::int64_t iz = v / sqxy;
    const std::int64_t p = v % sqxy;
    const std::int64_t iy = p / (nx + 1);
    const std::int64_t ix = p % (nx + 1);
    const double z = e + ef * static_cast<double>(iz);
    const double y = c + cd * static_cast<double>(iy);
    const double x = a + ab * static_cast<double>(ix);
    point = {x, y, z};
    for (std::size_t i = 0; i < 3; i++)
      geom(v - range_p[0], i) = point[i];
  }

  return geom;
}
//-----------------------------------------------------------------------------
dolfinx::graph::AdjacencyList<std::int32_t>
partition_graph(const MPI_Comm comm, int nparts,
                const dolfinx::graph::AdjacencyList<std::int64_t>& local_graph,
                std::int32_t num_ghost_nodes, bool ghosting)
{
  return dolfinx::graph::scotch::partitioner(dolfinx::graph::scotch::strategy::none, 0.01, 1010)(comm, nparts, local_graph,
                                      num_ghost_nodes, ghosting);
}
//-----------------------------------------------------------------------------
dolfinx::mesh::Mesh build_tet(MPI_Comm comm,
                              const std::array<std::array<double, 3>, 2>& p,
                              std::array<std::size_t, 3> n,
                              const dolfinx::mesh::GhostMode ghost_mode,
                              bool part_on_subset)
{
  dolfinx::common::Timer timer("Build BoxMesh");

  int rank = dolfinx::MPI::rank(comm);
  MPI_Comm nodecomm;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                      &nodecomm);
  int noderank = dolfinx::MPI::rank(nodecomm);
  int nodesize = dolfinx::MPI::size(nodecomm);
  MPI_Comm subcomm;

  // Number of cores per node to use for partitioning
  int num_cores_per_node = 1;
  
  if (part_on_subset)
  {
    int color = (noderank < num_cores_per_node) ? 0 : MPI_UNDEFINED;
    MPI_Comm_split(comm, color, rank, &subcomm);
  }
  else
  {
    MPI_Comm_dup(comm, &subcomm);
    noderank = 0;
  }

  xt::xtensor<double, 2> geom({0, 3});
  xt::xtensor<std::int64_t, 2> cells({0, 4});

  if (noderank < num_cores_per_node)
  {
    geom = create_geom(subcomm, p, n);

    const std::int64_t nx = n[0];
    const std::int64_t ny = n[1];
    const std::int64_t nz = n[2];
    const std::int64_t n_cells = nx * ny * nz;
    std::array range_c = dolfinx::MPI::local_range(
        dolfinx::MPI::rank(subcomm), n_cells, dolfinx::MPI::size(subcomm));
    const std::size_t cell_range = range_c[1] - range_c[0];
    LOG(INFO) << "Cell range: " << range_c[0] << "-" << range_c[1] << "\n";
    
    cells.resize({6 * cell_range, 4});

    // Create tetrahedra
    for (std::int64_t i = range_c[0]; i < range_c[1]; ++i)
    {
      const int iz = i / (nx * ny);
      const int j = i % (nx * ny);
      const int iy = j / nx;
      const int ix = j % nx;

      const std::int64_t v0 = iz * (nx + 1) * (ny + 1) + iy * (nx + 1) + ix;
      const std::int64_t v1 = v0 + 1;
      const std::int64_t v2 = v0 + (nx + 1);
      const std::int64_t v3 = v1 + (nx + 1);
      const std::int64_t v4 = v0 + (nx + 1) * (ny + 1);
      const std::int64_t v5 = v1 + (nx + 1) * (ny + 1);
      const std::int64_t v6 = v2 + (nx + 1) * (ny + 1);
      const std::int64_t v7 = v3 + (nx + 1) * (ny + 1);

      // Note that v0 < v1 < v2 < v3 < vmid
      xt::xtensor_fixed<std::int64_t, xt::xshape<6, 4>> c
          = {{v0, v1, v3, v7}, {v0, v1, v7, v5}, {v0, v5, v7, v4},
             {v0, v3, v2, v7}, {v0, v6, v4, v7}, {v0, v2, v6, v7}};
      std::size_t offset = 6 * (i - range_c[0]);
      xt::view(cells, xt::range(offset, offset + 6), xt::all()) = c;
    }
  }

  const std::function<const dolfinx::graph::AdjacencyList<std::int32_t>(
      MPI_Comm, int, int, const dolfinx::graph::AdjacencyList<std::int64_t>&,
      dolfinx::mesh::GhostMode)>
      partitioner
      = [&](MPI_Comm comm, int nparts, int tdim,
            const dolfinx::graph::AdjacencyList<std::int64_t>& cells,
            dolfinx::mesh::GhostMode mode) {
          dolfinx::graph::AdjacencyList<std::int32_t> procs(0);

          if (noderank < num_cores_per_node)
          {
            procs = dolfinx::mesh::partition_cells_graph(subcomm, nparts, tdim,
                                                         cells, mode, partition_graph);
          }
          return procs;
        };

  dolfinx::fem::CoordinateElement element(dolfinx::mesh::CellType::tetrahedron,
                                          1);
  auto [data, offset] = dolfinx::graph::create_adjacency_data(cells);
  return dolfinx::mesh::create_mesh(comm,
                                    dolfinx::graph::AdjacencyList<std::int64_t>(
                                        std::move(data), std::move(offset)),
                                    element, geom, ghost_mode, partitioner);
}

// Calculate number of vertices, edges, facets, and cells for any given level
// of refinement
constexpr std::tuple<std::int64_t, std::int64_t, std::int64_t, std::int64_t>
num_entities(int i, int j, int k, int nrefine)
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
std::int64_t num_pdofs(int i, int j, int k, int nrefine, int order)
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

std::shared_ptr<dolfinx::mesh::Mesh>
create_cube_mesh(MPI_Comm comm, std::size_t target_dofs, bool target_dofs_total,
                 std::size_t dofs_per_node, int order, bool part_on_subset)
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
    if (Nx > 100)
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

  auto mesh = std::make_shared<dolfinx::mesh::Mesh>(
      build_tet(comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {Nx, Ny, Nz},
                dolfinx::mesh::GhostMode::none, part_on_subset));

  if (dolfinx::MPI::rank(mesh->mpi_comm()) == 0)
  {
    std::cout << "UnitCube (" << Nx << "x" << Ny << "x" << Nz
              << ") to be refined " << r << " times\n";
  }

  for (int i = 0; i < r; ++i)
  {
    mesh->topology_mutable().create_connectivity(3, 1);
    mesh = std::make_shared<dolfinx::mesh::Mesh>(
        dolfinx::refinement::refine(*mesh, false));
  }

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
    const std::int32_t num_edges = mesh->topology().index_map(1)->size_local();
    std::vector<std::int32_t> mesh_indices;
    std::vector<std::int8_t> mesh_tags;
    for (int i = 0; i < num_edges; ++i)
    {
      if (i % 2000 < nmarked)
      {
        mesh_indices.push_back(i);
        mesh_tags.push_back(1);
      }
    }
    dolfinx::mesh::MeshTags<std::int8_t> marker(mesh, 1, mesh_indices,
                                                mesh_tags);

    mesh->topology_mutable().create_connectivity(1, 1);
    meshi = std::make_shared<dolfinx::mesh::Mesh>(
        dolfinx::refinement::refine(*mesh, marker, false));

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

// Copyright (C) 2019 Chris N. Richardson and Garth N. Wells
// Licensed under the MIT License. See LICENSE file in the project
// root for full license information.

#include "mesh.h"
#include <dolfin/common/MPI.h>
#include <dolfin/common/types.h>
#include <dolfin/generation/BoxMesh.h>
#include <dolfin/geometry/Point.h>
#include <dolfin/mesh/DistributedMeshTools.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Partitioning.h>
#include <dolfin/refinement/refine.h>

namespace
{
// Calculate number of vertices for any given level of refinement
std::int64_t nvertices(int i, int j, int k, int nrefine)
{
  std::int64_t nv = (i + 1) * (j + 1) * (k + 1);
  std::int64_t earr[3] = {1, 3, 7};
  for (int r = 0; r < nrefine; ++r)
  {
    std::size_t ne = earr[0] * (i + j + k) + earr[1] * (i * j + j * k + k * i)
                     + earr[2] * i * j * k;
    nv += ne;
    earr[0] *= 2;
    earr[1] *= 4;
    earr[2] *= 8;
  }
  return nv;
}
} // namespace

std::shared_ptr<dolfin::mesh::Mesh> create_cube_mesh(MPI_Comm comm,
                                                     std::size_t target_dofs,
                                                     bool target_dofs_total,
                                                     std::size_t dofs_per_node)
{
  // Get number of processes
  const std::size_t num_processes = dolfin::MPI::size(comm);

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
    nc = nvertices(Nx, Nx, Nx, r);
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
        std::size_t diff = std::abs(nvertices(i, j, k, r) - N);
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

  auto mesh = std::make_shared<dolfin::mesh::Mesh>(
      dolfin::generation::BoxMesh::create(
          comm,
          {dolfin::geometry::Point(0.0, 0.0, 0.0),
           dolfin::geometry::Point(1.0, 1.0, 1.0)},
          {Nx, Ny, Nz}, dolfin::mesh::CellType::Type::tetrahedron,
          dolfin::mesh::GhostMode::none));

  if (dolfin::MPI::rank(mesh->mpi_comm()) == 0)
  {
    std::cout << "UnitCube (" << Nx << "x" << Ny << "x" << Nz
              << ") to be refined " << r << " times\n";
  }

  for (unsigned int i = 0; i != r; ++i)
    mesh = std::make_shared<dolfin::mesh::Mesh>(
        dolfin::refinement::refine(*mesh, false));

  return mesh;
}
//-----------------------------------------------------------------------------
std::shared_ptr<dolfin::mesh::Mesh> create_spoke_mesh(MPI_Comm comm,
                                                      std::size_t target_dofs,
                                                      bool target_dofs_total,
                                                      std::size_t dofs_per_node)
{

  int target = target_dofs / dofs_per_node;

  int mpi_size = dolfin::MPI::size(comm);

  if (!target_dofs_total)
    target *= mpi_size;

  // Parameters controlling shape
  int n = 17;       // number of spokes
  double r0 = 0.25; // inner radius of ring
  double r1 = 0.5;  // outer radius of ring

  double h0 = 1.2; // height (inner)
  double h1 = 1.0; // height (outer)

  int lspur = 6;     // number of elements in each spoke
  double l0 = 0.5;   // length of each element in spoke
  double dth = 0.15; // curl (angle increment) as spoke goes out
  double tap = 0.9;  // taper (fractional height decrease on each element)

  // Subdivision of a cube into 6 tetrahedra
  int cube[6][4] = {{0, 1, 2, 4}, {1, 2, 4, 5}, {2, 4, 5, 6},
                    {0, 2, 3, 4}, {6, 7, 4, 2}, {2, 3, 4, 7}};

  // Calculate number of points and cells (only on process 0)
  int npoints = 0;
  int ncells = 0;
  const int mpi_rank = dolfin::MPI::rank(comm);

  if (mpi_rank == 0)
  {
    npoints = n * 4 + n * lspur * 4;
    ncells = n * 6 + n * lspur * 6;
  }

  dolfin::EigenRowArrayXXd geom(npoints, 3);
  dolfin::EigenRowArrayXXi64 topo(ncells, 4);

  if (mpi_rank == 0)
  {
    int p = 0;
    int c = 0;

    // Add n 'cubes' to make a joined up ring.
    for (int i = 0; i < n; ++i)
    {
      // Get the points for current cube
      Eigen::Array<int, 8, 1> pts;
      for (int j = 0; j < 8; ++j)
        pts[j] = ((i * 4 + j) % (n * 4));

      // Add to topology
      for (int k = 0; k < 6; ++k)
      {
        for (int j = 0; j < 4; ++j)
          topo(c, j) = pts[cube[k][j]];
        ++c;
      }

      // Calculate the position of points
      double th = 2 * M_PI * i / n;
      geom.row(p) << r0 * cos(th), r0 * sin(th), h0;
      ++p;
      geom.row(p) << r0 * cos(th), r0 * sin(th), -h0;
      ++p;
      geom.row(p) << r1 * cos(th), r1 * sin(th), -h1;
      ++p;
      geom.row(p) << r1 * cos(th), r1 * sin(th), h1;
      ++p;
    }

    // Add spurs to ring
    for (int i = 0; i < n; ++i)
    {
      // Intermediate angle between two faces
      double th0 = 2 * M_PI * (i + .5) / n;

      // Starting points on outer edge of ring
      Eigen::Array<int, 8, 1> pts;
      pts << (i * 4 + 2) % (n * 4), (i * 4 + 3) % (n * 4),
          (i * 4 + 7) % (n * 4), (i * 4 + 6) % (n * 4), 0, 0, 0, 0;

      // Build each spur outwards
      for (int k = 0; k < lspur; ++k)
      {
        // Add new points
        for (int j = 0; j < 4; ++j)
        {
          pts[j + 4] = p;
          geom.row(p) = geom.row(pts[j]);
          geom(p, 0) += l0 * cos(th0 + k * dth);
          geom(p, 1) += l0 * sin(th0 + k * dth);
          geom(p, 2) *= pow(tap, k);
          ++p;
        }

        // Add new cells
        for (int m = 0; m < 6; ++m)
        {
          for (int j = 0; j < 4; ++j)
            topo(c, j) = pts[cube[m][j]];
          ++c;
        }

        // Outer face becomes inner face of next cube
        pts.head(4) = pts.tail(4);
      }
    }

    // Check geometric sizes and rescale
    geom.col(0) -= 0.9 * geom.col(0).minCoeff();
    double scaling = 0.9 * geom.col(0).maxCoeff();
    geom /= scaling;

    std::cout << "x range = " << geom.col(0).minCoeff() << " - "
              << geom.col(0).maxCoeff() << "\n";
    std::cout << "y range = " << geom.col(1).minCoeff() << " - "
              << geom.col(1).maxCoeff() << "\n";
    std::cout << "z range = " << geom.col(2).minCoeff() << " - "
              << geom.col(2).maxCoeff() << "\n";
  }

  auto mesh = std::make_shared<dolfin::mesh::Mesh>(
      dolfin::mesh::Partitioning::build_distributed_mesh(
          comm, dolfin::mesh::CellType::Type::tetrahedron, geom, topo, {},
          dolfin::mesh::GhostMode::none));

  mesh->create_entities(1);
  dolfin::mesh::DistributedMeshTools::number_entities(*mesh, 1);

  std::cout << "target:" << target << "\n";

  while (mesh->num_entities_global(0) + mesh->num_entities_global(1) < target)
  {

    mesh = std::make_shared<dolfin::mesh::Mesh>(
        dolfin::refinement::refine(*mesh, false));

    mesh->create_entities(1);
    dolfin::mesh::DistributedMeshTools::number_entities(*mesh, 1);

    std::cout << mesh->num_entities_global(0) << std::endl;
    std::cout << mesh->num_entities_global(1) << std::endl;
  }

  double fraction = (double)(target - mesh->num_entities_global(0))
                    / mesh->num_entities_global(1);

  if (mpi_rank == 0)
    std::cout << "Desired fraction=" << fraction << std::endl;

  // Estimate step needed to get desired refinement fraction
  // using some heuristics and bisection method
  int nmarked = pow(fraction, 1.6) * 2000;

  int lmark = nmarked / 2;
  int umark = nmarked * 3;
  if (umark > 2000)
    umark = 2000;

  std::shared_ptr<dolfin::mesh::Mesh> meshi;

  for (int k = 0; k < 5; ++k)
  {
    // Trial step

    dolfin::mesh::MeshFunction<bool> marker(mesh, 1, false);
    for (int i = 0; i < mesh->num_entities(1); ++i)
      marker[i] = (i % 2000 < nmarked);

    meshi = std::make_shared<dolfin::mesh::Mesh>(
        dolfin::refinement::refine(*mesh, marker, false));

    double actual_fraction
        = (double)(meshi->num_entities_global(0) - mesh->num_entities_global(0))
          / mesh->num_entities_global(1);

    if (mpi_rank == 0)
    {
      std::cout << "Edges marked = " << nmarked << "/2000\n";
      std::cout << "Step " << k
                << " achieved actual fraction = " << actual_fraction << "\n";
    }

    if (actual_fraction > fraction)
    {
      umark = nmarked;
      nmarked = (nmarked + lmark) / 2;
    }
    else
    {
      lmark = nmarked;
      nmarked = (nmarked + umark) / 2;
    }
  }

  return meshi;
}

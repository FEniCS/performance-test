// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
// Licensed under the MIT License. See LICENSE file in the project
// root for full license information.

#pragma once

#include <dolfin/common/MPI.h>
#include <dolfin/generation/BoxMesh.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/refinement/refine.h>
#include <memory>

namespace {
// Calculate number of vertices for any given level of refinement
std::int64_t nvertices(int i, int j, int k, int nrefine) {
  std::int64_t nv = (i + 1) * (j + 1) * (k + 1);
  std::int64_t earr[3] = {1, 3, 7};
  for (int r = 0; r < nrefine; ++r) {
    std::size_t ne = earr[0] * (i + j + k) + earr[1] * (i * j + j * k + k * i) +
                     earr[2] * i * j * k;
    nv += ne;
    earr[0] *= 2;
    earr[1] *= 4;
    earr[2] *= 8;
  }
  return nv;
}
} // namespace

std::shared_ptr<const dolfin::mesh::Mesh>
create_mesh(MPI_Comm comm, std::size_t target_dofs, bool target_dofs_total,
            std::size_t dofs_per_node) {
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
  while (nc < N) {
    ++Nx;
    if (Nx > 100) {
      Nx = 40;
      ++r;
    }
    nc = nvertices(Nx, Nx, Nx, r);
  }

  Ny = Nx;
  Nz = Nx;

  std::size_t i0 = Nx - 10;
  std::size_t mindiff = 1000000;
  for (std::size_t i = i0; i < i0 + 20; ++i) {
    for (std::size_t j = i - 5; j < i + 5; ++j) {
      for (std::size_t k = i - 5; k < i + 5; ++k) {
        std::size_t diff = std::abs(nvertices(i, j, k, r) - N);
        if (diff < mindiff) {
          mindiff = diff;
          Nx = i;
          Ny = j;
          Nz = k;
        }
      }
    }
  }

  auto mesh =
      std::make_shared<const dolfin::mesh::Mesh>(dolfin::generation::BoxMesh(
          comm, {{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}, {Nx, Ny, Nz},
          dolfin::mesh::CellType::tetrahedron, dolfin::mesh::GhostMode::none));

  if (dolfin::MPI::rank(mesh->mpi_comm()) == 0) {
    std::cout << "UnitCube (" << Nx << "x" << Ny << "x" << Nz
              << ") to be refined " << r << " times\n";
  }

  for (unsigned int i = 0; i != r; ++i)
    mesh = std::make_shared<const dolfin::mesh::Mesh>(
        dolfin::refinement::refine(*mesh, false));

  return mesh;
}

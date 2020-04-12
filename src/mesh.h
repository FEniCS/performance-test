// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
// Licensed under the MIT License. See LICENSE file in the project
// root for full license information.

#pragma once

#include <memory>
#include <mpi.h>

namespace dolfinx
{
namespace fem
{
class CoordinateElement;
}
namespace mesh
{
class Mesh;
}
} // namespace dolfinx

std::shared_ptr<dolfinx::mesh::Mesh>
create_cube_mesh(MPI_Comm comm, const dolfinx::fem::CoordinateElement& cmap,
                 std::size_t target_dofs, bool target_dofs_total,
                 std::size_t dofs_per_node);

std::shared_ptr<dolfinx::mesh::Mesh>
create_spoke_mesh(MPI_Comm comm, const dolfinx::fem::CoordinateElement& cmap,
                  std::size_t target_dofs, bool target_dofs_total,
                  std::size_t dofs_per_node);

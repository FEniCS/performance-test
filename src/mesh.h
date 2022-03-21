// Copyright (C) 2017 Chris N. Richardson and Garth N. Wells
// Licensed under the MIT License. See LICENSE file in the project
// root for full license information.

#pragma once

#include <memory>
#include <mpi.h>

namespace dolfinx::fem
{
class CoordinateElement;
}

namespace dolfinx::mesh
{
class Mesh;
}

dolfinx::mesh::Mesh create_cube_mesh(MPI_Comm comm, std::size_t target_dofs,
                                     bool target_dofs_total,
                                     std::size_t dofs_per_node,
                                     std::size_t max_cells_per_dim,
                                     int target_cells_per_rank, int order);

std::shared_ptr<dolfinx::mesh::Mesh>
create_spoke_mesh(MPI_Comm comm, std::size_t target_dofs,
                  bool target_dofs_total, std::size_t dofs_per_node);

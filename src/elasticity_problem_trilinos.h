// Copyright (C) 2021 Chris N. Richardson
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <dolfinx/fem/Function.h>
#include <dolfinx/la/Vector.h>
#include <memory>
#include <utility>

namespace dolfinx::mesh
{
class Mesh;
}

namespace elastic_trilinos
{

std::tuple<dolfinx::la::Vector<PetscScalar>,
           std::shared_ptr<dolfinx::fem::Function<PetscScalar>>,
           std::function<int(dolfinx::fem::Function<PetscScalar>&,
                             const dolfinx::la::Vector<PetscScalar>&)>>
problem(std::shared_ptr<dolfinx::mesh::Mesh> mesh);

} // namespace elastic

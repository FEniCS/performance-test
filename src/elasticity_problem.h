// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <dolfinx/fem/Function.h>
#include <dolfinx/la/Vector.h>
#include <memory>
#include <petscsys.h>
#include <utility>

namespace dolfinx::mesh
{
template <std::floating_point T>
class Mesh;
}

namespace elastic
{

std::tuple<std::shared_ptr<dolfinx::la::Vector<PetscScalar>>,
           std::shared_ptr<dolfinx::fem::Function<PetscScalar>>,
           std::function<int(dolfinx::fem::Function<PetscScalar>&,
                             const dolfinx::la::Vector<PetscScalar>&)>>
problem(std::shared_ptr<dolfinx::mesh::Mesh<double>> mesh, int order);

} // namespace elastic

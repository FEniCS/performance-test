// Copyright (C) 2017-2019 Chris N. Richardson and Garth N. Wells
//
// This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    MIT

#pragma once

#include <dolfinx/fem/Function.h>
#include <dolfinx/la/Vector.h>
#include <dolfinx/mesh/Mesh.h>
#include <memory>
#include <petscsys.h>
#include <utility>

namespace poisson
{

std::tuple<std::shared_ptr<dolfinx::la::Vector<PetscScalar>>,
           std::shared_ptr<dolfinx::fem::Function<PetscScalar>>,
           std::function<int(dolfinx::fem::Function<PetscScalar>&,
                             const dolfinx::la::Vector<PetscScalar>&)>>
problem(std::shared_ptr<dolfinx::mesh::Mesh> mesh, int order);

} // namespace poisson

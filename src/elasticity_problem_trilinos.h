// Copyright (C) 2021 Chris N. Richardson
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

using T = PetscScalar;

namespace dolfinx::mesh
{
template <std::floating_point T>
class Mesh;
}

namespace elasticity_trilinos
{

std::tuple<std::shared_ptr<dolfinx::la::Vector<T>>,
           std::shared_ptr<dolfinx::fem::Function<T>>,
           std::function<int(dolfinx::fem::Function<T>&,
                             const dolfinx::la::Vector<T>&)>>
problem(std::shared_ptr<dolfinx::mesh::Mesh<double>> mesh, int order);

} // namespace elastic_trilinos

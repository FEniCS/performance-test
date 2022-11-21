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
#include "cuda_allocator.h"

namespace cgpoisson
{

using cuda_vector = la::Vector<PetscScalar, CUDA::allocator<PetscScalar>>;

std::tuple<std::shared_ptr<cuda_vector>, std::shared_ptr<cuda_vector>,
           std::function<int(cuda_vector&, const cuda_vector&)>>
problem(std::shared_ptr<dolfinx::mesh::Mesh> mesh, int order,
        std::string scatterer);

} // namespace cgpoisson

# Copyright (C) 2017-2022 Chris N. Richardson and Garth N. Wells
#
# This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT

import basix.ufl
from ufl import (Coefficient, Identity, FunctionSpace, Mesh, TestFunction, TrialFunction,
                 dx, grad, inner, tetrahedron, tr)

# Elasticity parameters
E = 1.0e6
nu = 0.3
mu = E / (2.0 * (1.0 + nu))
lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
cell = tetrahedron

# Load namespace
ns = vars()

forms = []
for degree in range(1, 4):
    element = basix.ufl.element("Lagrange", "tetrahedron", degree, shape=(3, ))
    domain = Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3, )))
    space = FunctionSpace(domain, element)

    u, v = TrialFunction(space), TestFunction(space)
    f = Coefficient(space)

    def eps(v):
        return 0.5*(grad(v) + grad(v).T)

    def sigma(v):
        return 2.0*mu*eps(v) + lmbda*tr(eps(v))*Identity(3)

    # Add forms to namespace with names a1, a2, a3 etc.
    aname = 'a' + str(degree)
    Lname = 'L' + str(degree)
    ns[aname] = inner(sigma(u), eps(v))*dx
    ns[Lname] = inner(f, v)*dx

    del u, v, f
    forms += [ns[aname], ns[Lname]]

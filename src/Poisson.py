# Copyright (C) 2017-2022 Chris N. Richardson and Garth N. Wells
#
# This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT

import basix.ufl
from ufl import (Coefficient, FunctionSpace, TestFunction, TrialFunction, Mesh, action, ds,
                 dx, grad, inner, tetrahedron)

# Load namespace
ns = vars()

forms = []
for degree in range(1, 4):
    element = basix.ufl.element("Lagrange", "tetrahedron", degree)
    domain = Mesh(basix.ufl.element("Lagrange", "tetrahedron", 1, shape=(3,)))
    space = FunctionSpace(domain, element)

    u = TrialFunction(space)
    v = TestFunction(space)
    f = Coefficient(space)
    g = Coefficient(space)
    un = Coefficient(space)

    aname = 'a' + str(degree)
    Lname = 'L' + str(degree)
    Mname = 'M' + str(degree)

    # Insert into namespace so that the forms will be named a1, a2, a3 etc.
    ns[aname] = inner(grad(u), grad(v))*dx
    ns[Lname] = f*v*dx + g*v*ds
    ns[Mname] = action(ns[aname], un)

    # Delete, so that the forms will get unnamed args and coefficients
    # and default to v_0, v_1, w0, w1 etc.
    del u, v, f, g, un

    forms += [ns[aname], ns[Lname], ns[Mname]]

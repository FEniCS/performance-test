# Copyright (C) 2017-2022 Chris N. Richardson and Garth N. Wells
#
# This file is part of FEniCS-miniapp (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    MIT

from ufl import (Coefficient, FiniteElement, TestFunction, TrialFunction, ds,
                 dx, grad, inner, tetrahedron)

# Load namespace
ns = vars()

forms = []
for degree in range(1, 4):
    element = FiniteElement("Lagrange", tetrahedron, degree)

    u = TrialFunction(element)
    v = TestFunction(element)
    f = Coefficient(element)
    g = Coefficient(element)

    aname = 'a' + str(degree)
    Lname = 'L' + str(degree)

    # Insert into namespace so that the forms will be named a1, a2, a3 etc.
    ns[aname] = inner(grad(u), grad(v))*dx
    ns[Lname] = f*v*dx + g*v*ds

    # Delete, so that the forms will get unnamed args and coefficients
    # and default to v_0, v_1, w0, w1 etc.
    del u, v, f, g

    forms += [ns[aname], ns[Lname]]

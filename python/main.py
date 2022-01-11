# Copyright (C) 2017-2021 Chris N. Richardson and Garth N. Wells
# FEniCS Project
# SPDX-License-Identifier:    MIT

import sys
from mpi4py import MPI
from petsc4py import PETSc
import numpy as np

import dolfinx
import dolfinx.fem
from ufl import TestFunction, TrialFunction, grad, inner, dx, ds

from mesh import create_cube_mesh

def poisson_problem(mesh, order):
    V = dolfinx.fem.FunctionSpace(mesh, ("CG", order))
    u = TrialFunction(V)
    v = TestFunction(V)

    f = dolfinx.fem.Function(V)
    g = dolfinx.fem.Function(V)

    # Interpolate into f and g

    a = dolfinx.fem.form(inner(grad(u), grad(v))*dx)
    L = dolfinx.fem.form(f*v*dx + g*v*ds)

    def bound(x):
        return np.logical_or(np.isclose(x[0], 0.0), np.isclose(x[0], 1.0))

    bc = dolfinx.fem.dirichletbc(PETSc.ScalarType(0.0), dolfinx.fem.locate_dofs_geometrical(V, bound), V)

    A = dolfinx.fem.assemble_matrix(a, bcs=[bc])
    A.assemble()

    b = dolfinx.fem.assemble_vector(L)
    dolfinx.fem.apply_lifting(b, [a], bcs=[[bc]])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    dolfinx.fem.set_bc(b, [bc])

    u = dolfinx.fem.Function(V)

    solver = PETSc.KSP().create(MPI.COMM_WORLD)
    solver.setFromOptions()

    # Set matrix operator
    solver.setOperators(A)

    return (b, u, solver)


# FIXME: get from command line
# FIXME: set petsc options from command line
problem_type = "poisson"
scaling_type = "weak"
ndofs = 500000
order = 1

num_processes = MPI.COMM_WORLD.Get_size()
ndofs_per_node = 3 if problem_type == "elasticity" else 1

with dolfinx.common.Timer("Create mesh"):
    mesh = create_cube_mesh(MPI.COMM_WORLD, ndofs, (scaling_type=="strong"), ndofs_per_node, order)

if problem_type=="poisson":
    b, u, solver = poisson_problem(mesh, order)

if MPI.COMM_WORLD.Get_rank() == 0:
    num_dofs = u.function_space.dofmap.index_map.size_global * u.function_space.dofmap.index_map_bs
    tdim = mesh.topology.dim
    num_cells = mesh.topology.index_map(tdim).size_global
    print("----------------------------------------------------------------")
    print("Test problem summary")
#    print("  dolfinx version: " << DOLFINX_VERSION_STRING)
#    print("  dolfinx hash:    " << DOLFINX_VERSION_GIT)
#    print("  ufl hash:        " << UFC_SIGNATURE)
#    print("  petsc version:   " << petsc_version)
    print("  Problem type:    ", problem_type)
    print("  Scaling type:    ", scaling_type)
    print("  Num processes:   ", num_processes)
    print("  Num cells        ", num_cells)
    print("  Total degrees of freedom:             ", num_dofs )
    print("  Average degrees of freedom per process: ", num_dofs / MPI.COMM_WORLD.Get_size())
    print("----------------------------------------------------------------")



with dolfinx.common.Timer("PETSc solve"):
    solver.solve(b, u.vector)

dolfinx.list_timings(MPI.COMM_WORLD, [dolfinx.TimingType.wall])

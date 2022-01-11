# Copyright (C) 2022 Chris N. Richardson and Garth N. Wells
import dolfinx
import dolfinx.graph
import dolfinx.mesh

def num_entities(i, j, k, nrefine):
    nv = (i + 1) * (j + 1) * (k + 1)
    ne = 0
    nc = (i * j * k) * 6
    earr = [1, 3, 7]
    farr = [2, 12]
    for r in range(nrefine):
        ne = (earr[0] * (i + j + k) + earr[1] * (i * j + j * k + k * i)
              + earr[2] * i * j * k)
        nv += ne
        nc *= 8
        earr[0] *= 2
        earr[1] *= 4
        earr[2] *= 8
        farr[0] *= 4
        farr[1] *= 8

    ne = (earr[0] * (i + j + k) + earr[1] * (i * j + j * k + k * i)
          + earr[2] * i * j * k)
    nf = farr[0] * (i * j + j * k + k * i) + farr[1] * i * j * k

    return (nv, ne, nf, nc)


def num_pdofs(i, j, k, nrefine, order):
    nv, ne, nf, nc = num_entities(i, j, k, nrefine)

    if (order == 1):
        return nv
    elif (order == 2):
        return nv + ne
    elif (order == 3):
        return nv + 2 * ne + nf
    elif (order == 4):
        return nv + 3 * ne + 3 * nf + nc
    else:
        raise ValueError("Order not supported")


def create_cube_mesh(comm, target_dofs, target_dofs_total, dofs_per_node, order):

    num_processes = comm.Get_size()

    # Target total dofs
    if target_dofs_total:
        N = target_dofs / dofs_per_node
    else:
        N = target_dofs * num_processes / dofs_per_node

    # Get initial guess for Nx, Ny, Nz, r
    r = 0
    Nx = 1
    nc = 0
    while (nc < N):

        Nx += 1
        if (Nx > 100):
            Nx = 40
            r += 1

        nc = num_pdofs(Nx, Nx, Nx, r, order)

    Ny = Nx
    Nz = Nx

    i0 = Nx - 10
    mindiff = 1000000
    for i in range(i0, i0 + 20):
        for j in range(i - 5, i +5):
            for k in range(i - 5, i + 5):
                diff = abs(num_pdofs(i, j, k, r, order) - N)
                if (diff < mindiff):
                    mindiff = diff
                    Nx = i
                    Ny = j
                    Nz = k

    graph_part = dolfinx.graph.partitioner()
    cell_part = dolfinx.mesh.create_cell_partitioner(graph_part)
    mesh = dolfinx.mesh.create_box(
      comm, [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], [Nx, Ny, Nz],
      dolfinx.mesh.CellType.tetrahedron, dolfinx.mesh.GhostMode.none,
      cell_part)

    if (comm.Get_rank() == 0):
        print(f"UnitCube ({Nx}x{Ny}x{Nz}) to be refined {r} times")

    for i in range(r):
        mesh.topology.create_entities(1)
        mesh = dolfinx.mesh.refine(mesh, redistribute=False)

    return mesh

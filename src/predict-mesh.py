
def num_entities(i, j, k, nrefine):
    nv = (i + 1) * (j + 1) * (k + 1)
    ne = 0
    nc = (i * j * k) * 6
    earr = [1, 3, 7]
    farr = [2, 12]
    for r in range(nrefine):
        ne = earr[0] * (i + j + k) + earr[1] * (i * j + j * k + k * i) \
            + earr[2] * i * j * k
        nv += ne
        nc *= 8
        earr[0] *= 2
        earr[1] *= 4
        earr[2] *= 8
        farr[0] *= 4
        farr[1] *= 8

    ne = earr[0] * (i + j + k) + earr[1] * (i * j + j * k + k * i) \
        + earr[2] * i * j * k
    nf = farr[0] * (i * j + j * k + k * i) + farr[1] * i * j * k

    return (nv, ne, nf, nc)


def num_pdofs(i, j, k, nrefine, order):
    nv, ne, nf, nc = num_entities(i, j, k, nrefine)
    print(i, j, k, nrefine, nv, nc)
    return nv


num_processes = 2600*128
dofs_per_process = 520000
# Target total dofs
N = num_processes * dofs_per_process

r = 0
Nx_max = 300
Nx = 1
ndofs = 0
order = 1

while (ndofs < N):

    Nx += 1
    if (Nx > Nx_max):
        while(ndofs < N):
            r += 1
            ndofs = num_pdofs(Nx, Nx, Nx, r, order)

        while (ndofs > N):
            Nx -= 1
            ndofs = num_pdofs(Nx, Nx, Nx, r, order)

    ndofs = num_pdofs(Nx, Nx, Nx, r, order)

print(N, ndofs)

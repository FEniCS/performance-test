# GPU performance test codes for FEniCSx/DOLFINx

This directory contains an implementation of the Laplacian operator for
hexahedral cells using sum factorisation, which runs on AMD on NVIDIA
GPUs using HIP or CUDA. It can be run in parallel with MPI, and allows
scaling by choosing the number of degrees of freedom per process.

## Requirements

- FEniCSx/DOLFINx installation (development version of DOLFINx
  **required**)
- HIP or CUDA compiler
- Boost Program Options

## Building

Use cmake to build, by creating a `build` subdirectory and using
`cmake`, followed by `make`. It is necessary to choose between AMD and
NVIDIA builds, see the cmake options, below.

### CMake options

* `-Damd=ON` builds using HIP
* `-Dnvidia=ON` builds using CUDA
* `-DSCALAR_TYPE=float32` will build a 32-bit version

e.g. to build for NVIDIA with float32
```
mkdir build
cd build
cmake -Dnvidia=ON -DSCALAR_TYPE=float32 ..
make
```

## Running tests

Options for the test are:

- Number of degrees-of-freedom (`--ndofs`): per MPI process
- Order (`--order`): polynomial degree P (2-7)
- Quadrature mode (`--qmode`): quadrature mode (0 or 1), qmode=0 has P+1 points
   in each direction, qmode=1 has P+2 points in each direction
- Gauss quadrature (`--use_gauss`): use Gauss rather than GLL
   quadrature
- Number of repetitions (`--nreps`)
- Geometry perturbation (`--geom_perturb_fact`) Adds a random
   perturbation to the geometry, useful to check correctness
- Matrix comparison (`--mat_comp`) Compare solution with CSR matrix
   (only useable for small `ndofs`)
- Geometry batch size (`--batch_size`) Geometry precomputation size
   (defaults to all precomputed)


## Recommended test configuration

Suggested options for running the test are listed below.

Single-GPU basic test for correctness (small problem)
```
./mat_free --order=5 --perturb_geom_fact=0.1 --mat_comp --ndofs=5000
```

Single-GPU performance test (10M dofs)
```
./mat_free --order=6 --ndofs=10000000 --qmode=1 --use_gauss
```

Multi-GPU performance test (40M dofs)
```
mpirun -n 4 ./mat_free --order=6 --ndofs=10000000 --qmode=1 --use_gauss
```

## Interpreting the output

The dolfinx timers provide information about the CPU portion of the
code, which creates the mesh, e.g.
- `Build BoxMesh (hexahedra)`: time taken to build the initial mesh

The GPU performance is presented as the number of GigaDOFs processed per
second: e.g. `Mat-free action Gdofs/s: 3.88691`

The norms of the input and output vectors are also provided, which can
be checked against the matrix (CSR) implementation be using the
`--mat_comp` option. In this case the norm of the error should be around
machine precision, i.e. about 1e-15 for float64.

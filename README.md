# Scaling tests for FEniCS/DOLFIN

This repository contains solvers for testing the parallel performance
of DOLFIN and the underlying linear solvers. It tests elliptic
equations - Poisson equation and elasticity - in three dimensions.

Representative performance [data](performance.md) is provided for
reference.


## Building

The source of the tests is in `src/` directory.


### Requirements

- FEniCS/DOLFIN installation, with MPI, PETSc and HDF5 enabled
  (development version of DOLFIN required)
- PETSc installation (development version required)

### Compilation

In the `src/` directory:

1. Compile the UFL files using FFC:

        ffc -l dolfin *.ufl

2. Build the program:

        cmake .
        make


## Running tests

Options for the test are:

- Problem type (`--problem_type`): `poisson` or `elasticity`
- Scaling type (`--scaling_type`): `strong` (fixed problem size) or `weak`
  (fixed problem size per process)
- Number of degrees-of-freedom (`--ndofs`): total (in case of strong
  scaling) or per process (for weak scaling)
- File output (`--output`): `true` or `false` (IO performance depends
  heavily on the underlying filesystem)
- Data output directory (`--output_dir`): directory to write solution
  data to

Linear solver options are configured via PETSc command line options,
as shown below.


## Recommended test configuration

Suggested options for running tests are listed below. The options
include PETSc performance logging which is useful for assessing
performance.

### Elasticity

For elasticity, a conjugate gradient (CG) solver with a smoothed
aggregation algebraic multigrid (GAMG) preconditioner is recommended.
For a weak scaling test with 8 MPI processes and 500k
degrees-of-freedom per process:

```
mpirun -np 8 ./dolfin-scaling-test \
--problem_type elasticity \
--scaling_type weak \
--ndofs 500000 \
--petsc.log_view \
--petsc.ksp_view \
--petsc.ksp_type cg \
--petsc.ksp_rtol 1.0e-8 \
--petsc.pc_type gamg \
--petsc.pc_gamg_coarse_eq_limit 1000 \
--petsc.mg_levels_ksp_type chebyshev \
--petsc.mg_levels_pc_type jacobi \
--petsc.mg_levels_esteig_ksp_type cg \
--petsc.matptap_via scalable \
--petsc.options_left
```

For a weak scaling test, with 8 MPI processes and 10M
degrees-of-freedom in total:


```
mpirun -np 8 ./dolfin-scaling-test \
--problem_type elasticity \
--scaling_type strong \
--ndofs 10000000 \
--petsc.log_view \
--petsc.ksp_view \
--petsc.ksp_type cg \
--petsc.ksp_rtol 1.0e-8 \
--petsc.pc_type gamg \
--petsc.pc_gamg_coarse_eq_limit 1000 \
--petsc.mg_levels_ksp_type chebyshev \
--petsc.mg_levels_pc_type jacobi \
--petsc.mg_levels_esteig_ksp_type cg \
--petsc.matptap_via scalable \
--petsc.options_left
```

### Poisson

For the Poisson equation, a conjugate gradient (CG) solver with a
classical algebraic multigrid (BoomerAMG) preconditioner is
recommended.  For a weak scaling test with 8 MPI processes and 500k
degrees-of-freedom per process:

```
mpirun -np 8 ./dolfin-scaling-test \
--problem_type poisson \
--scaling_type weak \
--ndofs 500000 \
--petsc.log_view \
--petsc.ksp_view \
--petsc.ksp_type cg \
--petsc.ksp_rtol 1.0e-8 \
--petsc.pc_type hypre \
--petsc.pc_hypre_type boomeramg \
--petsc.pc_hypre_boomeramg_strong_threshold 0.5 \
--petsc.options_left
```
For a strong scaling test, with 8 MPI processes and 10M
degrees-of-freedom in total:
```
mpirun -np 8 ./dolfin-scaling-test \
--problem_type poisson \
--scaling_type strong \
--ndofs 10000000 \
--petsc.log_view \
--petsc.ksp_view \
--petsc.ksp_type cg \
--petsc.ksp_rtol 1.0e-8 \
--petsc.pc_type hypre \
--petsc.pc_hypre_type boomeramg \
--petsc.pc_hypre_boomeramg_strong_threshold 0.5 \
--petsc.options_left
```

## Reference performance data

Reference performance data is provided [here](performance.md) to help
in assessing performance on a given system.

## Authors and license

The tests have been developed by Chris N. Richardson
(<chris@bpi.cam.ac.uk>) and Garth N. Wells (<gnw20@cam.ac.uk>).

The code is covered by the MIT license. See LICENSE.md.

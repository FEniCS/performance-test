# Performance test codes for FEniCS/DOLFIN-X

This repository contains solvers for testing the parallel performance
of DOLFIN-X and the underlying linear solvers. It tests elliptic
equations - Poisson equation and elasticity - in three dimensions.

Representative performance [data](performance.md) is provided for
reference (Work in Progress).


## Building

The source of the tests is in `src/` directory.


### Requirements

- FEniCS/DOLFIN-X installation, with MPI, PETSc and HDF5 enabled
  (development version of dolfinx **required**)
- PETSc installation (development version **required**)

The development versions of dolfinx and PETSc have a number of fixes
that are necessary to run the tests successfully, and for performance.

### Compilation

In the `src/` directory, build the program:

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
(single hyphen) as shown below.


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
mpirun -np 8 ./dolfinx-scaling-test \
--problem_type elasticity \
--scaling_type weak \
--ndofs 500000 \
-log_view \
-ksp_view \
-ksp_type cg \
-ksp_rtol 1.0e-8 \
-pc_type gamg \
-pc_gamg_coarse_eq_limit 1000 \
-mg_levels_ksp_type chebyshev \
-mg_levels_pc_type jacobi \
-mg_levels_esteig_ksp_type cg \
-matptap_via scalable \
-options_left
```

For a strong scaling test, with 8 MPI processes and 10M
degrees-of-freedom in total:


```
mpirun -np 8 ./dolfinx-scaling-test \
--problem_type elasticity \
--scaling_type strong \
--ndofs 10000000 \
-log_view \
-ksp_view \
-ksp_type cg \
-ksp_rtol 1.0e-8 \
-pc_type gamg \
-pc_gamg_coarse_eq_limit 1000 \
-mg_levels_ksp_type chebyshev \
-mg_levels_pc_type jacobi \
-mg_levels_esteig_ksp_type cg \
-matptap_via scalable \
-options_left
```

### Poisson

For the Poisson equation, a conjugate gradient (CG) solver with a
classical algebraic multigrid (BoomerAMG) preconditioner is
recommended.  For a weak scaling test with 8 MPI processes and 500k
degrees-of-freedom per process:

```
mpirun -np 8 ./dolfinx-scaling-test \
--problem_type poisson \
--scaling_type weak \
--ndofs 500000 \
-log_view \
-ksp_view \
-ksp_type cg \
-ksp_rtol 1.0e-8 \
-pc_type hypre \
-pc_hypre_type boomeramg \
-pc_hypre_boomeramg_strong_threshold 0.5 \
-options_left
```
For a strong scaling test, with 8 MPI processes and 10M
degrees-of-freedom in total:
```
mpirun -np 8 ./dolfinx-scaling-test \
--problem_type poisson \
--scaling_type strong \
--ndofs 10000000 \
-log_view \
-ksp_view \
-ksp_type cg \
-ksp_rtol 1.0e-8 \
-pc_type hypre \
-pc_hypre_type boomeramg \
-pc_hypre_boomeramg_strong_threshold 0.5 \
-options_left
```

## Reference performance data

Reference performance data is provided [here](performance.md) to help
in assessing performance on a given system.


## Test status

The code is tested on CircleCI.
[![CircleCI](https://circleci.com/gh/FEniCS/performance-test.svg?style=svg)](https://circleci.com/gh/FEniCS/performance-test)

## Authors and license

The tests have been developed by Chris N. Richardson
(<chris@bpi.cam.ac.uk>) and Garth N. Wells (<gnw20@cam.ac.uk>).

The code is covered by the MIT license. See LICENSE.md.

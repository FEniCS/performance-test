# Performance test codes for FEniCSx/DOLFINx

This repository contains solvers for testing the parallel performance of
DOLFINx and the underlying linear solvers. It tests elliptic equations
- Poisson equation and elasticity - in three dimensions.

Representative performance data is available at
https://fenics.github.io/performance-test-results/.

[![FEniCS Performance Test CI](https://github.com/FEniCS/performance-test/workflows/FEniCS%20Performance%20Test%20CI/badge.svg)](https://github.com/FEniCS/performance-test/actions?query=branch%3Amain)

## Building

The source of the tests is in `src/` directory.


### Requirements

- FEniCSx/DOLFINx installation (development version of DOLFINx
  **required**)
- PETSc installation
- Boost Program Options


### Compilation

In the `src/` directory, build the program:

        cmake .
        make


## Running tests

Options for the test are:

- Problem type (`--problem_type`): `poisson` or `elasticity`
- Scaling type (`--scaling_type`): `strong` (fixed problem size) or
  `weak` (fixed problem size per process)
- Number of degrees-of-freedom (`--ndofs`): total (in case of strong
  scaling) or per process (for weak scaling)
- Order (`--order`): polynomial order (1, 2, or 3) - only on cube mesh,
  defaults to 1.
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
For a weak scaling test with 8 MPI processes and 500k degrees-of-freedom
per process:

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
-pc_hypre_boomeramg_strong_threshold 0.7 \
-pc_hypre_boomeramg_agg_nl 4 \
-pc_hypre_boomeramg_agg_num_paths 2 \
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
-pc_hypre_boomeramg_strong_threshold 0.7 \
-pc_hypre_boomeramg_agg_nl 4 \
-pc_hypre_boomeramg_agg_num_paths 2 \
-options_left
```

## Interpreting the output

The default loglevel diagnostic messages from DOLFINx will be present, and if `-log_view` is specified, there will be a performance profile from PETSc. There's also a "Test problem summary" summarizing the test parameters and environment to aid with reproducibility. Finally, there's a table labeled "Summary of timings" that contains various times (in units of seconds) of interest, the parts that are explicit to this test are labeled `ZZZ`. We elaborate on some:

- `ZZZ Create Mesh`: Create the mesh to be used as the spatial discretisation of the domain in the FE problem
- `ZZZ Create facets and facet->cell connectivity`: Compute the topology connectivity of the mesh's graph, i.e. compute the relationship between which cells are connected to each facet.
- `ZZZ FunctionSpace`: Create the function space in which the finite element method solution will be sought along with appropriate index maps for each degree of freedom and their relationship with the mesh.
- `ZZZ Assemble`: Encompassing timer for:
  - `ZZZ Create boundary conditions`: Find the mesh’s topological indices and corresponding degree of freedom indices on which to impose boundary data in a strong Dirichlet sense.
  - `ZZZ Create RHS function`: This is the step computing the function $f$ in the cases where $\nabla^2u=-f$ (Poisson) and $\nabla\cdot u=-f$ (elasticity, i.e. elastostatics in this case).
  - `ZZZ Assemble matrix`: Assemble the finite element matrix $A$ underlying finite element formulation, such that we seek to later solve $A\vec{x}=\vec{b}$.
  - `ZZZ Assemble vector`: Assemble the right-hand-side vector $\vec{b}$.
- `ZZZ Solve`: Compute the solution of the linear system. This is typically the dominant stage taking the greatest computational effort.
- `ZZZ Output`: Postprocess and potentially output (with `--output`) results to disk.


## Reference performance data

Reference performance data is provided [here](performance.md) to help
in assessing performance on a given system.


## Authors and license

The tests have been developed by Chris N. Richardson
(<chris@bpi.cam.ac.uk>) and Garth N. Wells (<gnw20@cam.ac.uk>).

The code is covered by the MIT license. See LICENSE.md.

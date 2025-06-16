# DOLFINx benchmark

This benchmark tests the performance of an unstructured grid finite element
solver. It solves the Poisson equation on a mesh of hexahedral cells
using a matrix-free method. Low- and high-degree finite elements bases
are supported. Being matrix-free and supporting high-degree finite
elements makes this benchmark suitable for CPU and GPU architectures.

Parallel communication between nodes/devices used MPI. The finite
element implementation uses sum factorisation.

## Status

Under development.

## Maintainers

[@chrisrichardson](https://www.github.com/chrisrichardson)

## Overview

### Main code/library

[DOLFINx](https://github.com/fenics/dolfinx)

### Architectures

CPU (in progress), GPU.

### Languages and programming models

C++, CUDA, HIP, MPI.

### 'Dwarfs'

Unstructured grids, dense linear algebra.

## Building

The benchmark can be built using Spack or manually using CMake.

### Spack

A Spack package is provided in `spack/`. To view the package options:
```bash
spack repo add ./spack
spack info bench-dolfinx
```
The benchmark builds an executable `bench_dolfinx`.


### CMake

The benchmark depends on the library
[DOLFINx](https://github.com/fenics/dolfinx) and can be built using
CMake. See the benchmark Spack package
[file](spack/packages/bench-dolfinx/package.py) and the Spack
dependencies for a comprehensive list of dependencies.

When building the benchmark using CMake, the following
benchmark-specific CMake options are available:
* `-DHIP_ARCH=[target]` builds using HIP for GPU architecture `[target]`
* `-DCUDA_ARCH=[target]` builds using CUDA for GPU architecture `[target]`
* `-DSCALAR_TYPE=float32` will build a 32-bit version

## Command line options

TODO: The program should give the options with the `-h` option.
```bash
bench_dolfinx -h
```

Once this is the case, remove the below.


Options for the test are:

- Quadrature mode (`--qmode`): quadrature mode (0 or 1), `qmode=0 `has
  `P+1` points in each direction, `qmode=1` has `P+2` points in each
  direction
- Gauss quadrature (`--use_gauss`): use Gauss rather than GLL quadrature
- Number of repetitions (`--nreps`)
- Geometry perturbation (`--geom_perturb_fact`) Adds a random
  perturbation to the geometry, useful to check correctness
- Matrix comparison (`--mat_comp`) Compare solution with CSR matrix
  (only useable for small `ndofs`)


## Benchmarks

TODO

### Correctness tests

TODO


### Performance tests

TODO

### Suggested performance test configuration


### Figures of merit

TODO


### Old text

The DOLFINx timers provide information about the CPU portion of the
code, which creates the mesh, e.g.
- `Build BoxMesh (hexahedra)`: time taken to build the initial mesh

The GPU performance is presented as the number of GigaDOFs processed per
second: e.g. `Mat-free action Gdofs/s: 3.88691`

The norms of the input and output vectors are also provided, which can
be checked against the matrix (CSR) implementation be using the
`--mat_comp` option. In this case the norm of the error should be around
machine precision, i.e. about 1e-15 for float64.

### OLD: Recommended test configuration

Suggested options for running the test are listed below.

Single-GPU basic test for correctness (small problem)
```bash
bench_dolfinx --order=5 --perturb_geom_fact=0.1 --mat_comp --ndofs=5000
```

Single-GPU performance test (10M dofs)
```bash
bench_dolfinx --order=6 --ndofs=10000000 --qmode=1 --use_gauss
```

Multi-GPU performance test (40M dofs)
```bash
mpirun -n 4 bench_dolfinx --order=6 --ndofs=10000000 --qmode=1 --use_gauss
```

## License

The benchmark code is released under the MIT license.

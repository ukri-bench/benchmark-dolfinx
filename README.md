# DOLFINx benchmark

This benchmark tests the performance of an unstructured grid finite
element solver. It solves the Poisson equation on a mesh of hexahedral
cells using a matrix-free method. Low- and high-degree finite elements
bases are supported. Being matrix-free and supporting high-degree finite
elements makes this benchmark suitable for CPU and GPU architectures.
The finite element implementation uses sum factorisation.

Parallel communication between nodes/devices uses MPI.

## Status

Under development.

## Maintainers

[@chrisrichardson](https://www.github.com/chrisrichardson),
[@garth-wells](https://www.github.com/garth-wells)

## Overview

### Main code/library

[DOLFINx](https://github.com/fenics/dolfinx)

### Architectures

CPU (in progress), GPU.

### Languages and programming models

C++, CUDA, HIP, MPI.

### Seven 'dwarfs'

- [x] Dense linear algebra
- [x] Sparse linear algebra
- [ ] Spectral methods
- [ ] N-body methods
- [ ] Structured grids
- [x] Unstructured grids
- [ ] Monte Carlo

## Building

The benchmark can be built using Spack or manually using CMake.

### Spack

A Spack package is provided in the repository
https://github.com/ukri-bench/spack-packages. To view the package
options:

```bash
spack repo add --name bench_pkgs https://github.com/ukri-bench/spack-packages.git bench_pkgs
spack info bench-dolfinx
```

Options are used tp specify CPU and GPU (AMD or CUDA) builds. The
benchmark builds an executable `bench_dolfinx`.

### CMake

The benchmark depends on the library
[DOLFINx](https://github.com/fenics/dolfinx) and can be built using
CMake. See the benchmark Spack package
[file](spack/packages/bench-dolfinx/package.py) and the Spack
dependencies for a comprehensive list of dependencies.

When building the benchmark using CMake, the following
benchmark-specific CMake options are available:
* `-DHIP_ARCH=[target]` builds using HIP for the specific GPU architecture `[target]`
* `-DCUDA_ARCH=[target]` builds using CUDA for the specific GPU architecture `[target]`

## Command line options

The program lists the available options with the `-h` option.
```bash
bench_dolfinx -h
```

## Benchmarks

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
machine precision, i.e. about 1e-15 for `float64`.

### OLD: Recommended test configuration

Suggested options for running the test are listed below.

Single-GPU basic test for correctness (small problem)
```bash
bench_dolfinx --degree=5 --perturb_geom_fact=0.1 --mat_comp --ndofs=5000
```

Single-GPU performance test (10M dofs)
```bash
bench_dolfinx --degree=6 --ndofs=10000000 --qmode=1 --use_gauss
```

Multi-GPU performance test (40M dofs)
```bash
mpirun -n 4 bench_dolfinx --degree=6 --ndofs=10000000 --qmode=1 --use_gauss
```

## License

The benchmark code is released under the MIT license.

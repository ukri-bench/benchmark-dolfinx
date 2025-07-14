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

Options are used to pecify CPU and GPU (AMD or CUDA) builds. The
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

Single device (single GPU or single CPU):
```bash
bench_dolfinx --float=64 --degree=6 --ndofs=10000000 --use_gauss --qmode=1
```
This runs the benchmark with $10^{7}$ degrees-of-freedom per device
(`--ndofs`) using 64-bit floats (`--float`), finite element basis degree
6 (`--degree`), Gauss-Legendre quadrature (`--use_gauss`)  and 'full'
quadrature (`--qmode=1`).

Multiple MPI ranks (1 MPI rank per logical GPU):
```bash
mpiexec -n 12 bench_dolfinx --float=64 --degree=6 --ndofs=10000000 --qmode=1 --use_gauss
```

### Suggested performance test configuration

```bash
mpiexec -n 12 bench_dolfinx --float=64 --degree=6 --ndofs=10000000 --qmode=1 --use_gauss
```

### Figures of merit

Performance data is written to the output JSON file. The figures of
merit include:

1. Throughout (degrees-of-freedom per second)
   - First iteration
   - Average of subsequent iterations
2. Geometry computation
3. Mesh creation

For GPU execution, (3) is executed on the CPU. All other operations are
executed on the GPU.

#### Degrees-of-freedom per second

Using the options, ...., report the degrees-of-freedom per second for:
- `ndofs * np = 5e9`

## License

The benchmark code is released under the MIT license.

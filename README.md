# DOLFINx benchmark

[![Spack build and run](https://github.com/ukri-bench/benchmark-dolfinx/actions/workflows/spack.yml/badge.svg)](https://github.com/ukri-bench/benchmark-dolfinx/actions/workflows/spack.yml)
[![Test benchmark](https://github.com/ukri-bench/benchmark-dolfinx/actions/workflows/ci.yml/badge.svg)](https://github.com/ukri-bench/benchmark-dolfinx/actions/workflows/ci.yml)

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
spack repo add --name fenics https://github.com/FEniCS/spack-fenics.git fenics
spack info bench-dolfinx
```

Options are used to specify CPU and GPU (AMD or CUDA) builds, e.g. `+cuda cuda_arch=80` or `+rocm amdgpu_target=gfx90a`. The
benchmark builds an executable `bench_dolfinx`.

### CMake

The benchmark depends on the library
[DOLFINx](https://github.com/fenics/dolfinx) v0.10.0 and can be built using
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

Compare against the same computation by assembling a matrix:

`bench_dolfinx --mat_comp --ndofs_global=10000 --degree=3`

This test can be used to verify the matrix-free GPU algorithm is
giving the same results as an assembled matrix. Because the matrix is
assembled on CPU, it can be very slow for large problems and high
polynomial degree. Recommended settings are 10000 global dofs, and
degree 3. Results should be the same in parallel with `mpirun`.
The console output should report `Norm of error` with a small (machine
precision) number e.g. for float64 about `1e-15`.

### Performance tests

The following tests are recommended. A problem size of at least 10M
dofs is needed to overcome the GPU launch latency. Problem size
per-GPU can be increased until the memory limit is reached. The number
of repetitions defaults to 1000.

Single-GPU performance test (10M dofs)
```bash
bench_dolfinx --float=64 --degree=6 --ndofs=10000000
```

Multi-GPU performance test (10M dofs per GPU)
```bash
mpirun -n 4 bench_dolfinx --float=64  --degree=6 --ndofs=10000000
```

Adding the `--cg` flag will also test additional `axpy` and global reduce
on every iteration. The `--float=32` flag will test at 32-bit
precision. Changing the `--degree` will affect the balance of
computation and communication (e.g. degree 6 is more computationally
efficient, but results in more inter-GPU data transfer on each iteration).

### Figures of merit

The main *Figure of Merit* (FoM) is the computational throughput in
GDoF/s. The throughput represents the amount of useful computation
that is done by the operator (or Conjugate Gradient) algorithm, and is
reported for the whole system. Thus, to get the throughput per GPU,
divide by the number of GPUs used. It is printed at the end of each
run, and can also be saved in a JSON file by adding the `--json
filename.json` flag.

## License

The benchmark code is released under the MIT license.

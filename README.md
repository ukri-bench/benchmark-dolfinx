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
[file](https://github.com/ukri-bench/spack-packages/blob/main/spack_pkgs/spack_repo/ukri_bench/packages/bench_dolfinx/package.py) and the Spack dependencies for a comprehensive list of dependencies.

When building the benchmark using CMake, the following
benchmark-specific CMake options are available:
* `-DHIP_ARCH=[target]` builds using HIP for the specific GPU architecture `[target]`
* `-DCUDA_ARCH=[target]` builds using CUDA for the specific GPU architecture `[target]`

### Potential compilation issues

- The dependency `basix` requires BLAS libraries. On Cray systems using `cray-libsci` these need to be specified to `cmake`.
This is encoded in the spack recipe at [https://github.com/FEniCS/spack-fenics/blob/e8b5e9fdd299889b4cb6209559de04b9289c20ab/spack_repo/fenics/packages/fenics_basix/package.py].

- The version of `mdspan.hpp` distributed in `basix` v0.10.0 is not compatible with CUDA 13. A patch is available at [https://github.com/FEniCS/spack-fenics/blob/07b9fd0dfd3d878c383ed8cba9e2a10fa52b478a/spack_repo/fenics/packages/fenics_basix/mdspan.patch], which should be applied if using CUDA 13.0 or higher.

- A C++20 compiler capable of handling `std::format` is required. On some systems, it is necessary to explicitly pass this to `nvcc` or `hipcc` through a command line argument, e.g. `--gcc-toolchain=/opt/rh/gcc-toolset-13/root/usr`.

- On Cray systems it may be necessary to explicity give the MPI path in the `CMakeLists.txt`.

## Running the benchmarks

### Selecting a GPU device and binding to CPU NUMA regions

The `bench_dolfinx` code is designed to run on one CPU MPI rank per
GPU device. In order to correctly map devices to cores, it is
usually necessary to include a GPU binding script between `mpirun`
and `bench_dolfinx`. There is an example of [how to do this on
LUMI-G](https://docs.lumi-supercomputer.eu/runjobs/scheduled-jobs/distribution-binding/#gpu-binding)
for ROCm and
[CSD3](https://docs.hpc.cam.ac.uk/hpc/user-guide/a100.html) for
CUDA. It is also important to bind the CPU cores to the correct NUMA
regions, as also described in these links. Additionally, MPI must
have GPU support enabled (e.g. `export MPICH_GPU_SUPPORT_ENABLED=1`
for Cray-MPICH).

### Running on a HPC system

The benchmark will often be run on a HPC system using a batch queueing system, such as SLURM.
A typical submission script is shown below:

```
#!/bin/bash
#SBATCH -p partition
#SBATCH --nodes=16
#SBATCH --gpus=64
#SBATCH --exclusive
#SBATCH --job-name=benchmark
#SBATCH --ntasks-per-node=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00

source /project/spack/share/spack/setup-env.sh
spack env activate bench10
module load libfabric/1.22.0

# Check correctness compared to matrix
srun -N ${SLURM_NNODES} -n ${SLURM_NTASKS} ./select_gpu ./bench_dolfinx --nreps=1 --mat_comp --ndofs_global=100000 --degree=3 --json mat_comp-${SLURM_NNODES}.json

# Run Q3 problem with 300M dofs/device
srun --mem-bind=local --cpu-bind=map_cpu:0,72,144,216 -N ${SLURM_NNODES} -n ${SLURM_NTASKS} ./select_gpu ./bench_dolfinx --ndofs=300000000 --degree=3 --cg --json Q3-300M.json
# Run Q6 problem with 500M dofs/device
srun --mem-bind=local --cpu-bind=map_cpu:0,72,144,216 -N ${SLURM_NNODES} -n ${SLURM_NTASKS} ./select_gpu ./bench_dolfinx --ndofs=500000000 --degree=6 --cg --json Q6-500M.json
```
See [examples] for example input and output files.

### Command line options

The program lists the available options with the `-h` option.
```bash
bench_dolfinx -h
```

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

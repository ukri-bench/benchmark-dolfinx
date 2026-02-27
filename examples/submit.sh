#!/bin/bash
#SBATCH -p partition
#SBATCH --nodes=16
#SBATCH --gpus=64
#SBATCH --exclusive
#SBATCH --job-name=benchmark
#SBATCH --ntasks-per-node=4
#SBATCH --hint=nomultithread
#SBATCH --time=00:20:00

source /projects/spack/share/spack/setup-env.sh
spack env activate bench10
module load libfabric/1.22.0

# Check correctness compared to matrix
srun -N ${SLURM_NNODES} -n ${SLURM_NTASKS} ./select_gpu ./bench_dolfinx --nreps=1 --mat_comp --ndofs_global=100000 --degree=3 --json mat_comp-${SLURM_NNODES}.json

srun --mem-bind=local --cpu-bind=map_cpu:0,72,144,216 -N ${SLURM_NNODES} -n ${SLURM_NTASKS} ./select_gpu_2 ./bench_dolfinx --ndofs=300000000 --degree=3 --cg --json=Q3-300M.json
srun --mem-bind=local --cpu-bind=map_cpu:0,72,144,216 -N ${SLURM_NNODES} -n ${SLURM_NTASKS} ./select_gpu_2 ./bench_dolfinx --ndofs=500000000 --degree=6 --cg --json=Q6-500M.json

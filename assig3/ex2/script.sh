#!/bin/bash
#SBATCH -o /home/hpc/t1221/lu26xuk/ass3/benchmarks/output/MPI1/myjob.%j.%N.out
#SBATCH -D /home/hpc/t1221/lu26xuk/ass3/benchmarks/src
#SBATCH -J Benchmarks
#SBATCH --get-user-env
#SBATCH --clusters=mpp3
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=none 
#SBATCH --mail-user=bene.kucis@tum.de
#SBATCH --time=03:00:00
source /etc/profile.d/modules.sh
mpiexec ./IMB-MPI1
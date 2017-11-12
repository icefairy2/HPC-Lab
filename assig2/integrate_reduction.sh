#!/bin/bash
#SBATCH -o /home/hpc/t1221/t1221ak/Ex2/integrateReduction.%j.%N.out 
#SBATCH -D /home/hpc/t1221/t1221ak/Ex2
#SBATCH -J IntegrateReduction 
#SBATCH --get-user-env 
#SBATCH --clusters=mpp3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
source /etc/profile.d/modules.sh
srun ./integrate_reduction

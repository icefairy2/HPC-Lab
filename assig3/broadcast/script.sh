#!/bin/bash
#SBATCH -o /home/hpc/t1221/lu26xut/broadcast/myjob.%j.%N.out
#SBATCH -D /home/hpc/t1221/lu26xut/broadcast
#SBATCH -J Broadcast
#SBATCH --get-user-env
#SBATCH --clusters=mpp3
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
source /etc/profile.d/modules.sh
mpiexec ./broadcast
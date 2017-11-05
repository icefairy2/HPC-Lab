#!/bin/bash
#SBATCH -o /home/hpc/t1221/lu26xut/a1/ex1/myjob.%j.%N.out 
#SBATCH -D /home/hpc/t1221/lu26xut/a1/ex1
#SBATCH -J PrintHostnames 
#SBATCH --get-user-env 
#SBATCH --clusters=mpp3
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
source /etc/profile.d/modules.sh
srun ./ex1.out

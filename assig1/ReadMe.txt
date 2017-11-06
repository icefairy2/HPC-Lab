Answers to the questions can be found in ass1_notes.mdown

The "hello world" files consist of:
helloworld.c 
hw-script.sh
myjob.7033.mpp3r01c02s02.out

The Files for part 3 are:
Makefile.gauss
gauss.cpp
gauss.optrpt


The Files for part 4 are:
Makefile.dgemm
dgemm.cpp
myjob.7259.mpp3r01c04s10.out


Compilation instructions:
make -f Makefile.gauss
make -f Makefile.dgemm 

Run the programs in an interactive shell:
salloc 
srun ./gauss 
or
srun ./dgemm

(or use mpiexec)

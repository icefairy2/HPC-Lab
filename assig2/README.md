# How to Run

### Assignment
All answers to the questions can be found in `assig2_notes.mdown`

### Presentation
The presentation can be found at the `assig2-pres.pdf`.

### Logs and Charts
All logs can be found in the logs folder. Any charts drawn are embedded in the Assignment Notes.

### Exercise 1
Compile exercise 1 files with:
`icpc integrate_reduction.cpp -o integrate_reduction -std=c++11 -qopenmp`

To vary the number of threads before execution:
`export OMP_NUM_THREADS=4`

### Exercise 2
Notes in the mdown file as listed above

### Exercise 3
Compile exercise 3 with
`make -f Makefile.qsort`

### Exercise 4
Compile exercise 4 with: `make -f Makefile.dmg`

To alter values of M N and K and MC use the Makefile.dmg. Unfortunately libxsmm generates kernels for hardcoded values. `dgemm.cpp` needs to be changed before correct results can be obtained after each change of M,N,K. The easy way is to use one of the already generated kernels (definitions in kernels.h) and use it in the microkernel function of dgemm.cpp.
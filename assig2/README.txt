Compile exercise 1 files with:
icpc integrate_reduction.cpp -o integrate_reduction -std=c++11 -qopenmp
To vary the number of threads before execution:
export OMP_NUM_THREADS=4
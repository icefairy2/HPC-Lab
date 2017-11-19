#define NDEBUG TRUE
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>

#include <immintrin.h>
#include <omp.h>
#include "kernels/kernels.h"

#include "Stopwatch.h"

//#define min(i,j) ((i)<(j) ? (i) : (j))

// macros for better readability
#define A(i,j) A[ (j)*lda + (i)]
#define B(i,j) B[ (j)*ldb + (i)]
#define C(i,j) C[ (j)*ldc + (i)]

/////////////////////////////////
/* Used in libxsmm kernels */
long libxsmm_num_total_flops = 0;

/** Syntax: M x N, ld S
 *  A M x N sub-block of of S x something matrix
 *  in column-major layout.
 *  That is C_ij := C[j*S + i], i=0,...,M-1,  j=0,...,N-1
 */
void dgemm(double* A, double* B, double* C, int S) {
  for (int n = 0; n < S; ++n) {
    for (int m = 0; m < S; ++m) {
      for (int k = 0; k < S; ++k) {
        C[n*S + m] += A[k*S + m] * B[n*S + k];
      }
    }
  }
}

void pack_B(double* B, double* B_pack, int S) {
	for (int j = 0; j < S; j++) {
		for (int i = 0; i < K; i++) {
			B_pack[i] = B[i];
		}
		B += S;
		B_pack += K;
	}
}

void pack_A(double* A, double* A_pack, int S) {
	for (int j = 0; j < K; j++) {
		for (int i = 0; i < MC; i++) {
			A_pack[i] = A[i];
		}
		A += S;
		A_pack += MC;
	}
}

void pack_A_v2(double *A, double *A_pack, int S) {
  double *A_ref;
  for(int i=0; i<MC; i+=M) {
    A_ref = A;
    for (int j=0; j<K; j++) {
      for (int k=0; k<M; k++) {
        A_pack[k] = A_ref[k+i];
      }
      A_pack+=M;
      A_ref +=S;
    }
  }
}

/**
  * A: M x K, ld M
  * B: K x N, ld K
  * C: M x N, ld S
  */
void microkernel(double* A, double* B, double* C, int S) {
  kernel_M128_N128_K128_S2048_knl(A, B, C);
}

/**
 * A: MC x K, ld S
 * B:  K x S, ld K
 * C: MC x S, ld S
 */
void GEBP(double* A, double* B, double* C, double* A_pack, int S, int threadsPerTeam)
{
	
	// pack A
  pack_A_v2(A, A_pack, S); // We pack A in such a manner that the access to every MxK block is contiguous
	pack_A(A, A_pack, S);
	#pragma omp parallel for num_threads(threadsPerTeam)
	for (int n = 0; n < S; n += N) {
		//for (int m = 0; m < MC; m += M) { // For using mr number of rows
      //int tid = omp_get_thread_num();
      //printf("Level %d: number of threads in the team %d- %d\n",
      //          2,  tid,omp_get_num_threads());
      // microkernel(&A_pack[m], &B[n*K], &C[n*S + m], S);
      microkernel(A_pack, &B[n*K], &C[n*S], S);
    //}
	}
}

/**
 * A: S x K, ld S
 * B: K x S, ld S
 * C: S x S, ld S
 */
void GEPP(double* A, double* B, double* C, double** A_pack, double* B_pack, int S, int nTeams, int threadsPerTeam)
{
	//pack B
	pack_B(B, B_pack, S);
	#pragma omp parallel for num_threads(nTeams)
	for (int m = 0; m < S; m += MC) {
		
		int tid = omp_get_thread_num();
    //printf("Level %d: number of threads in the team %d- %d\n",
    //         1,  tid,omp_get_num_threads());
		
		
		//GEBP(&A[m], B, &C[m], A_pack[0], S, threadsPerTeam); // works for square no packing
		//GEBP(&A[m], B_pack, &C[m], A_pack[0], S, threadsPerTeam); // works
		
		GEBP(&A[m], B_pack, &C[m], A_pack[tid], S, threadsPerTeam); // parallelism
	}
}

/**
 * A: S x S, ld S
 * B: S x S, ld S
 * C: S x S, ld S
 */
void GEMM(double* A, double* B, double* C, double** A_pack, double* B_pack, int S, int nTeams = 1, int threadsPerTeam = 1) {
  for (int k = 0; k < S; k+=K) {
	  GEPP(&A[k*S], &B[k], C, A_pack, B_pack, S, nTeams, threadsPerTeam);
  }
}


int main(int argc, char** argv) {
  bool test = true;
  int threadsPerTeam = 1;
  int nRepeat = 10;
  int S=256;
  if (argc <= 1) {
    printf("Usage: dgemm <S> <test> <threadsPerTeam> <repetitions>");
    return -1;
  }
  if (argc > 1) {
    S = atoi(argv[1]);
  }
  if (argc > 2) {
    test = atoi(argv[2]) != 0;
  }
  if (argc > 3) {
    threadsPerTeam = atoi(argv[3]);
  }
  if (argc > 4) {
    nRepeat = atoi(argv[4]);
  }

  omp_set_nested(1);

  int nThreads, nTeams;
  #pragma omp parallel
  #pragma omp master
  {
    nThreads = omp_get_num_threads(); 
  }
  threadsPerTeam = std::min(threadsPerTeam, nThreads);
  nTeams = nThreads / threadsPerTeam;
  
  printf("Num threads allowed: %d\n", omp_get_max_threads());
  printf("nTeams: %d\n", nTeams);
  printf("threadsPerTeam: %d\n", threadsPerTeam);
  
  /** Allocate memory */
  double* A, *B, *C, *A_test, *B_test, *C_test, **A_pack, *B_pack, *C_aux;
  
  posix_memalign(reinterpret_cast<void**>(&A),      ALIGNMENT, S*S*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&B),      ALIGNMENT, S*S*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&C),      ALIGNMENT, S*S*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&A_test), ALIGNMENT, S*S*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&B_test), ALIGNMENT, S*S*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&C_test), ALIGNMENT, S*S*sizeof(double));

  posix_memalign(reinterpret_cast<void**>(&A_pack), ALIGNMENT, nTeams*sizeof(double*));
  for (int t = 0; t < nTeams; ++t) {
    posix_memalign(reinterpret_cast<void**>(&A_pack[t]), ALIGNMENT, MC*K*sizeof(double));
  }
  posix_memalign(reinterpret_cast<void**>(&B_pack), ALIGNMENT,  K*S*sizeof(double));

  #pragma omp parallel for
  for (int j = 0; j < S; ++j) {
    for (int i = 0; i < S; ++i) {
      /*A[j*S + i] = i + j;
      B[j*S + i] = (S-i) + (S-j);*/
      A[j*S + i] = j*S + i;
      B[j*S + i] = j*S + i;
      C[j*S + i] = 0.0;
    }
  }
  memcpy(A_test, A, S*S*sizeof(double));
  memcpy(B_test, B, S*S*sizeof(double));
  memset(C_test, 0, S*S*sizeof(double));
  
  /** Check correctness of optimised dgemm */
  if (test) {
    #pragma noinline
    {
      dgemm(A_test, B_test, C_test, S);
      GEMM(A, B, C, A_pack, B_pack, S, nTeams, threadsPerTeam);
    }

    double error = 0.0;
    for (int i = 0; i < S*S; ++i) {
      double diff = C[i] - C_test[i];
      error += diff*diff;
    }
    error = sqrt(error);
    if (error > std::numeric_limits<double>::epsilon()) {
      printf("Optimised DGEMM is incorrect. Error: %e\n", error);
      return -1;
    }
  }

  Stopwatch stopwatch;
  double time;

  /** Test performance of microkernel */

  stopwatch.start();
  for (int i = 0; i < 10000; ++i) {
    #pragma noinline
    microkernel(A, B, C, S);
    __asm__ __volatile__("");
  }
  time = stopwatch.stop();
  printf("Microkernel: %lf ms, %lf GFLOP/s\n", time*1.0e3, 10000*2.0*M*N*K/time * 1.0e-9);

  /** Test performance of GEBP */

  stopwatch.start();
  for (int i = 0; i < nRepeat; ++i) {
    #pragma noinline
    GEBP(A, B, C, A_pack[0], S, threadsPerTeam);
    __asm__ __volatile__("");
  }
  time = stopwatch.stop();
  printf("GEBP: %lf ms, %lf GFLOP/s\n", time*1.0e3, nRepeat*2.0*MC*S*K/time * 1.0e-9);

  /** Test performance of optimised GEMM */

  stopwatch.start();
  for (int i = 0; i < nRepeat; ++i) {
    #pragma noinline
    GEMM(A, B, C, A_pack, B_pack, S, nTeams, threadsPerTeam);
    __asm__ __volatile__("");
  }
  time = stopwatch.stop();
  printf("GEMM: %lf ms, %lf GFLOP/s\n", time * 1.0e3, nRepeat*2.0*S*S*S/time * 1.0e-9);
  
  /** Clean up */
  /*
  free(A); free(B); free(C);
  free(A_test); free(B_test); free(C_test);
  for (int t = 0; t < nTeams; ++t) {
    free(A_pack[t]);
  }
  free(A_pack); free(B_pack);
*/
  return 0;
}

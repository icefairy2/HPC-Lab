extern long libxsmm_num_total_flops;
#include <stdio.h>
void kernel_M2_N2_K4_S8_noarch(const double* A, const double* B, double* C) {
#pragma message ("LIBXSMM KERNEL COMPILATION WARNING: compiling arch-independent gemm kernel in: " __FILE__)
  unsigned int l_m = 0;
  unsigned int l_n = 0;
  unsigned int l_k = 0;

  for ( l_n = 0; l_n < 2; l_n++ ) {
    for ( l_k = 0; l_k < 4; l_k++ ) {
      #pragma simd
      for ( l_m = 0; l_m < 2; l_m++ ) {
        C[(l_n*8)+l_m] += A[(l_k*2)+l_m] * B[(l_n*4)+l_k];
      }
    }
  }
#ifndef NDEBUG
#ifdef _OPENMP
#pragma omp atomic
#endif
libxsmm_num_total_flops += 32;
#endif
}


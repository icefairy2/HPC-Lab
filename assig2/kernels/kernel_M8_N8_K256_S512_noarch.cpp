extern long libxsmm_num_total_flops;

void kernel_M8_N8_K256_S512_noarch(const double* A, const double* B, double* C) {
#pragma message ("LIBXSMM KERNEL COMPILATION WARNING: compiling arch-independent gemm kernel in: " __FILE__)
  unsigned int l_m = 0;
  unsigned int l_n = 0;
  unsigned int l_k = 0;

  for ( l_n = 0; l_n < 8; l_n++ ) {
    for ( l_k = 0; l_k < 256; l_k++ ) {
      #pragma simd
      for ( l_m = 0; l_m < 8; l_m++ ) {
        C[(l_n*512)+l_m] += A[(l_k*8)+l_m] * B[(l_n*256)+l_k];
      }
    }
  }
#ifndef NDEBUG
#ifdef _OPENMP
#pragma omp atomic
#endif
libxsmm_num_total_flops += 32768;
#endif
}


#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>

#include "Stopwatch.h"

#include "immintrin.h" //avx512



// macros for better readability
#define A(i,j) A[ (j)*lda + (i)]
#define B(i,j) B[ (j)*ldb + (i)]
#define C(i,j) C[ (j)*ldc + (i)]

// for blocking
#define mc 256
#define kc 128
#define nb 1000

#define min(i,j) ((i)<(j) ? (i) : (j))


void dgemm(double* A, double* B, double* C) {
  for (int n = 0; n < N; ++n) {
    for (int m = 0; m < M; ++m) {
      for (int k = 0; k < K; ++k) {
        C[n*M + m] += A[k*M + m] * B[n*K + k];
      }
    }
  }
}

/**
 * Kernel computing C += A*B
 * Uses intrinsics for computation
 */
void inner512_8x8(int K_, double* A, int lda, double* B, int ldb, double* C, int ldc) {
	
	// vector registers
	__m512d
	c_00_10_20_30_40_50_60_70_v,
	c_01_11_21_31_41_51_61_71_v,
	c_02_12_22_32_42_52_62_72_v,
	c_03_13_23_33_43_53_63_73_v,
	c_04_14_24_34_44_54_64_74_v,
	c_05_15_25_35_45_55_65_75_v,
	c_06_16_26_36_46_56_66_76_v,
	c_07_17_27_37_47_57_67_77_v,
	a_0k_1k_2k_3k_4k_5k_6k_7k_v,
	b_k0_v2, b_k1_v2, b_k2_v2, b_k3_v2,
	b_k4_v2, b_k5_v2, b_k6_v2, b_k7_v2,
	tmp_v;
	
	c_00_10_20_30_40_50_60_70_v = _mm512_setzero_pd();
	c_01_11_21_31_41_51_61_71_v = _mm512_setzero_pd();
	c_02_12_22_32_42_52_62_72_v = _mm512_setzero_pd();
	c_03_13_23_33_43_53_63_73_v = _mm512_setzero_pd();
	c_04_14_24_34_44_54_64_74_v = _mm512_setzero_pd();
	c_05_15_25_35_45_55_65_75_v = _mm512_setzero_pd();
	c_06_16_26_36_46_56_66_76_v = _mm512_setzero_pd();
	c_07_17_27_37_47_57_67_77_v = _mm512_setzero_pd();
	
	int b = 0;
	
	#pragma vector always
	for (int k = 0; k < K_; k+=4) {
		
		a_0k_1k_2k_3k_4k_5k_6k_7k_v = _mm512_load_pd( (double *) &A(0,k));
		
		/** we left this here to show that we tried to use packing for A and B*/
		//use for packing B
		//b_k0_v2 = _mm512_set1_pd(B[b]);
		//b_k1_v2 = _mm512_set1_pd(B[b+1]);
		//b_k2_v2 = _mm512_set1_pd(B[b+2]);
		//b_k3_v2 = _mm512_set1_pd(B[b+3]);
		//b_k4_v2 = _mm512_set1_pd(B[b+4]);
		//b_k5_v2 = _mm512_set1_pd(B[b+5]);
		//b_k6_v2 = _mm512_set1_pd(B[b+6]);
		//b_k7_v2 = _mm512_set1_pd(B[b+7]);	
	    //b+=8;
		
		b_k0_v2 = _mm512_set1_pd(B(k,0));
		b_k1_v2 = _mm512_set1_pd(B(k,1));
		b_k2_v2 = _mm512_set1_pd(B(k,2));
		b_k3_v2 = _mm512_set1_pd(B(k,3));
		b_k4_v2 = _mm512_set1_pd(B(k,4));
		b_k5_v2 = _mm512_set1_pd(B(k,5));
		b_k6_v2 = _mm512_set1_pd(B(k,6));
		b_k7_v2 = _mm512_set1_pd(B(k,7));
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k0_v2);
		c_00_10_20_30_40_50_60_70_v = _mm512_add_pd(tmp_v, c_00_10_20_30_40_50_60_70_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k1_v2);
		c_01_11_21_31_41_51_61_71_v = _mm512_add_pd(tmp_v, c_01_11_21_31_41_51_61_71_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k2_v2);
		c_02_12_22_32_42_52_62_72_v = _mm512_add_pd(tmp_v, c_02_12_22_32_42_52_62_72_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k3_v2);
		c_03_13_23_33_43_53_63_73_v = _mm512_add_pd(tmp_v, c_03_13_23_33_43_53_63_73_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k4_v2);
		c_04_14_24_34_44_54_64_74_v = _mm512_add_pd(tmp_v, c_04_14_24_34_44_54_64_74_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k5_v2);
		c_05_15_25_35_45_55_65_75_v = _mm512_add_pd(tmp_v, c_05_15_25_35_45_55_65_75_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k6_v2);
		c_06_16_26_36_46_56_66_76_v = _mm512_add_pd(tmp_v, c_06_16_26_36_46_56_66_76_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k7_v2);
		c_07_17_27_37_47_57_67_77_v = _mm512_add_pd(tmp_v, c_07_17_27_37_47_57_67_77_v);
		
		a_0k_1k_2k_3k_4k_5k_6k_7k_v = _mm512_load_pd( (double *) &A(0,k+1));
		
		b_k0_v2 = _mm512_set1_pd(B(k+1,0));
		b_k1_v2 = _mm512_set1_pd(B(k+1,1));
		b_k2_v2 = _mm512_set1_pd(B(k+1,2));
		b_k3_v2 = _mm512_set1_pd(B(k+1,3));
		b_k4_v2 = _mm512_set1_pd(B(k+1,4));
		b_k5_v2 = _mm512_set1_pd(B(k+1,5));
		b_k6_v2 = _mm512_set1_pd(B(k+1,6));
		b_k7_v2 = _mm512_set1_pd(B(k+1,7));
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k0_v2);
		c_00_10_20_30_40_50_60_70_v = _mm512_add_pd(tmp_v, c_00_10_20_30_40_50_60_70_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k1_v2);
		c_01_11_21_31_41_51_61_71_v = _mm512_add_pd(tmp_v, c_01_11_21_31_41_51_61_71_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k2_v2);
		c_02_12_22_32_42_52_62_72_v = _mm512_add_pd(tmp_v, c_02_12_22_32_42_52_62_72_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k3_v2);
		c_03_13_23_33_43_53_63_73_v = _mm512_add_pd(tmp_v, c_03_13_23_33_43_53_63_73_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k4_v2);
		c_04_14_24_34_44_54_64_74_v = _mm512_add_pd(tmp_v, c_04_14_24_34_44_54_64_74_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k5_v2);
		c_05_15_25_35_45_55_65_75_v = _mm512_add_pd(tmp_v, c_05_15_25_35_45_55_65_75_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k6_v2);
		c_06_16_26_36_46_56_66_76_v = _mm512_add_pd(tmp_v, c_06_16_26_36_46_56_66_76_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k7_v2);
		c_07_17_27_37_47_57_67_77_v = _mm512_add_pd(tmp_v, c_07_17_27_37_47_57_67_77_v);
		
		a_0k_1k_2k_3k_4k_5k_6k_7k_v = _mm512_load_pd( (double *) &A(0,k+2));
		
		b_k0_v2 = _mm512_set1_pd(B(k+2,0));
		b_k1_v2 = _mm512_set1_pd(B(k+2,1));
		b_k2_v2 = _mm512_set1_pd(B(k+2,2));
		b_k3_v2 = _mm512_set1_pd(B(k+2,3));
		b_k4_v2 = _mm512_set1_pd(B(k+2,4));
		b_k5_v2 = _mm512_set1_pd(B(k+2,5));
		b_k6_v2 = _mm512_set1_pd(B(k+2,6));
		b_k7_v2 = _mm512_set1_pd(B(k+2,7));
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k0_v2);
		c_00_10_20_30_40_50_60_70_v = _mm512_add_pd(tmp_v, c_00_10_20_30_40_50_60_70_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k1_v2);
		c_01_11_21_31_41_51_61_71_v = _mm512_add_pd(tmp_v, c_01_11_21_31_41_51_61_71_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k2_v2);
		c_02_12_22_32_42_52_62_72_v = _mm512_add_pd(tmp_v, c_02_12_22_32_42_52_62_72_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k3_v2);
		c_03_13_23_33_43_53_63_73_v = _mm512_add_pd(tmp_v, c_03_13_23_33_43_53_63_73_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k4_v2);
		c_04_14_24_34_44_54_64_74_v = _mm512_add_pd(tmp_v, c_04_14_24_34_44_54_64_74_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k5_v2);
		c_05_15_25_35_45_55_65_75_v = _mm512_add_pd(tmp_v, c_05_15_25_35_45_55_65_75_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k6_v2);
		c_06_16_26_36_46_56_66_76_v = _mm512_add_pd(tmp_v, c_06_16_26_36_46_56_66_76_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k7_v2);
		c_07_17_27_37_47_57_67_77_v = _mm512_add_pd(tmp_v, c_07_17_27_37_47_57_67_77_v);
		
		a_0k_1k_2k_3k_4k_5k_6k_7k_v = _mm512_load_pd( (double *) &A(0,k+3));
		
		b_k0_v2 = _mm512_set1_pd(B(k+3,0));
		b_k1_v2 = _mm512_set1_pd(B(k+3,1));
		b_k2_v2 = _mm512_set1_pd(B(k+3,2));
		b_k3_v2 = _mm512_set1_pd(B(k+3,3));
		b_k4_v2 = _mm512_set1_pd(B(k+3,4));
		b_k5_v2 = _mm512_set1_pd(B(k+3,5));
		b_k6_v2 = _mm512_set1_pd(B(k+3,6));
		b_k7_v2 = _mm512_set1_pd(B(k+3,7));
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k0_v2);
		c_00_10_20_30_40_50_60_70_v = _mm512_add_pd(tmp_v, c_00_10_20_30_40_50_60_70_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k1_v2);
		c_01_11_21_31_41_51_61_71_v = _mm512_add_pd(tmp_v, c_01_11_21_31_41_51_61_71_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k2_v2);
		c_02_12_22_32_42_52_62_72_v = _mm512_add_pd(tmp_v, c_02_12_22_32_42_52_62_72_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k3_v2);
		c_03_13_23_33_43_53_63_73_v = _mm512_add_pd(tmp_v, c_03_13_23_33_43_53_63_73_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k4_v2);
		c_04_14_24_34_44_54_64_74_v = _mm512_add_pd(tmp_v, c_04_14_24_34_44_54_64_74_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k5_v2);
		c_05_15_25_35_45_55_65_75_v = _mm512_add_pd(tmp_v, c_05_15_25_35_45_55_65_75_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k6_v2);
		c_06_16_26_36_46_56_66_76_v = _mm512_add_pd(tmp_v, c_06_16_26_36_46_56_66_76_v);
		
		tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		tmp_v = _mm512_mul_pd(tmp_v, b_k7_v2);
		c_07_17_27_37_47_57_67_77_v = _mm512_add_pd(tmp_v, c_07_17_27_37_47_57_67_77_v);
		
	}
	
	_mm512_store_pd(&C(0,0), c_00_10_20_30_40_50_60_70_v);
	_mm512_store_pd(&C(0,1), c_01_11_21_31_41_51_61_71_v);
	_mm512_store_pd(&C(0,2), c_02_12_22_32_42_52_62_72_v);
	_mm512_store_pd(&C(0,3), c_03_13_23_33_43_53_63_73_v);
	_mm512_store_pd(&C(0,4), c_04_14_24_34_44_54_64_74_v);
	_mm512_store_pd(&C(0,5), c_05_15_25_35_45_55_65_75_v);
	_mm512_store_pd(&C(0,6), c_06_16_26_36_46_56_66_76_v);
	_mm512_store_pd(&C(0,7), c_07_17_27_37_47_57_67_77_v);
}

/**
* We're not using the packing methods, but we left those in the code
* to show that we tried to use them
*/

/** pack A in contiguous memory */
void packA(int K_, double* A, int lda, double* A_dest) {
	
	//#pragma vector always
	for (int k = 0; k < K_; k++) {
		double
		 *a_ij_ptr = &A(0,k);
		 
		 *A_dest++ = *a_ij_ptr;
		 *A_dest++ = *(a_ij_ptr+1);
		 *A_dest++ = *(a_ij_ptr+2);
		 *A_dest++ = *(a_ij_ptr+3);
		 *A_dest++ = *(a_ij_ptr+4);
		 *A_dest++ = *(a_ij_ptr+5);
		 *A_dest++ = *(a_ij_ptr+6);
		 *A_dest++ = *(a_ij_ptr+7);
	}
}

/** pack B in contiguous memory */
void packB(int K_, double* B, int ldb, double* B_dest) {
	double 
	*b_k0_ptr = &B(0,0), *b_k1_ptr = &B(0,1),
	*b_k2_ptr = &B(0,2), *b_k3_ptr = &B(0,3),
	*b_k4_ptr = &B(0,4), *b_k5_ptr = &B(0,5),
	*b_k6_ptr = &B(0,6), *b_k7_ptr = &B(0,7);
	

	for (int k = 0; k < K_; k++) {
		
		*B_dest++ = *b_k0_ptr++;
		*B_dest++ = *b_k1_ptr++;
		*B_dest++ = *b_k2_ptr++;
		*B_dest++ = *b_k3_ptr++;
		*B_dest++ = *b_k4_ptr++;
		*B_dest++ = *b_k5_ptr++;
		*B_dest++ = *b_k6_ptr++;
		*B_dest++ = *b_k7_ptr++;
	}
	
}


void innerKernel(int M_, int N_, int K_, double* A, int lda, double* B, int ldb, double* C, int ldc, int first) {
	
	/** for packing */
	//double packedA[M_*K_];
	//double packedB[kc * nb];


	for (int n = 0; n < N_; n+=8) {
		/** for packing */
		//if (first) packB(K_, &B(0,n), ldb, &packedB[n*K_]);
		for (int m = 0; m < M_; m+=8) {
		  /** for packing */
		  //if (n == 0) packA(K_, &A(m,0), lda, &packedA[m*K_]);
		  /** we used this to test packing */
		  //inner512_8x8(K_, &packedA[m*K_], 8, &B(0,n),ldb ,&C(m,n), ldc);	  
		  //inner512_8x8(K_, &A(m,0), lda, &packedB[n*K_],K_ ,&C(m,n), ldc);
		  
		  /** Kernel call without packing*/
		  inner512_8x8(K_, &A(m,0), lda, &B(0,n),ldb ,&C(m,n), ldc);
		}
	}
}

void dgemm_opt(double* A, double* B, double* C) {
  
  int kb, mb;
  int lda = M;
  int ldb = K;
  int ldc = M;
  
  for (int k = 0; k < K; k+=kc) {
	  kb = min(K-k, kc);
	  for (int m = 0; m < M; m+=mc) {
		  mb = min(M-m, mc);
		  innerKernel(mb, N, kb, &A(m,k), lda, &B(k,0), ldb, &C(m,0), ldc, m==0);
	  } 
	 }
}

int main(int argc, char** argv) {
  int repetitions = 10000;
  if (argc > 1) {
    repetitions = atoi(argv[1]);
  }
  
  /** Allocate memory */
  double* A, *B, *C, *A_test, *B_test, *C_test;
  
  posix_memalign(reinterpret_cast<void**>(&A),      ALIGNMENT, M*K*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&B),      ALIGNMENT, K*N*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&C),      ALIGNMENT, M*N*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&A_test), ALIGNMENT, M*K*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&B_test), ALIGNMENT, K*N*sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&C_test), ALIGNMENT, M*N*sizeof(double));

  for (int j = 0; j < K; ++j) {
    for (int i = 0; i < M; ++i) {
      A[j*M + i] = i + j;
    }
  }
  for (int j = 0; j < N; ++j) {
    for (int i = 0; i < K; ++i) {
      B[j*K + i] = (K-i) + (N-j);
    }
  }
  memset(C, 0, M*N*sizeof(double));
  memcpy(A_test, A, M*K*sizeof(double));
  memcpy(B_test, B, K*N*sizeof(double));
  memset(C_test, 0, M*N*sizeof(double));
  
  /** Check correctness of optimised dgemm */
  #pragma noinline
  {
    dgemm(A, B, C);
    dgemm_opt(A_test, B_test, C_test);
  }

  double error = 0.0;
  for (int i = 0; i < M*N; ++i) {
    double diff = C[i] - C_test[i];
    error += diff*diff;
  }
  error = sqrt(error);
  if (error > std::numeric_limits<double>::epsilon()) {
    printf("Optimised DGEMM is incorrect. Error: %e\n", error);
    return -1;
  }
  
  /** Test performance of optimised dgemm */
  
  #pragma noinline
  dgemm_opt(A, B, C);
  
  Stopwatch stopwatch;
  stopwatch.start();

  #pragma noinline
  for (int r = 0; r < repetitions; ++r) {
    dgemm_opt(A, B, C);
    __asm__ __volatile__("");
  }
  
  double time = stopwatch.stop();
  printf("%lf ms, %lf GFLOP/s\n", time * 1.0e3, repetitions*2.0*M*N*K/time * 1.0e-9);
  
  /** Clean up */
  
  free(A); free(B); free(C);
  free(A_test); free(B_test); free(C_test);

  return 0;
}

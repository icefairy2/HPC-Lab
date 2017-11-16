#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>

#include <immintrin.h>
#include <omp.h>

#include "Stopwatch.h"

//#define min(i,j) ((i)<(j) ? (i) : (j))

// macros for better readability
#define A(i,j) A[ (j)*lda + (i)]
#define B(i,j) B[ (j)*ldb + (i)]
#define C(i,j) C[ (j)*ldc + (i)]

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

/**
 * A: M x K, ld M
 * B: K x N, ld K
 * C: M x N, ld S
 */
void microkernel(double* A, double* B, double* C, int S) {
  /*
  int lda = 64;
  int ldb = 64;
  int ldc = 64;
  
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
	b_k4_v2, b_k5_v2, b_k6_v2, b_k7_v2;
	
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
	for (int k = 0; k < S; k+=4) {  // S here is kb
		
		a_0k_1k_2k_3k_4k_5k_6k_7k_v = _mm512_load_pd( (double *) &A(0,k));
		
		
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
		
	
		c_00_10_20_30_40_50_60_70_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k0_v2, c_00_10_20_30_40_50_60_70_v);
		c_01_11_21_31_41_51_61_71_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k1_v2, c_01_11_21_31_41_51_61_71_v);
		c_02_12_22_32_42_52_62_72_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k2_v2, c_02_12_22_32_42_52_62_72_v);
		c_03_13_23_33_43_53_63_73_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k3_v2, c_03_13_23_33_43_53_63_73_v);
		c_04_14_24_34_44_54_64_74_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k4_v2, c_04_14_24_34_44_54_64_74_v);
		c_05_15_25_35_45_55_65_75_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k5_v2, c_05_15_25_35_45_55_65_75_v);
		c_06_16_26_36_46_56_66_76_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k6_v2, c_06_16_26_36_46_56_66_76_v);
		c_07_17_27_37_47_57_67_77_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k7_v2, c_07_17_27_37_47_57_67_77_v);
		
		a_0k_1k_2k_3k_4k_5k_6k_7k_v = _mm512_load_pd( (double *) &A(0,k+1));
		b_k0_v2 = _mm512_set1_pd(B(k+1,0));
		b_k1_v2 = _mm512_set1_pd(B(k+1,1));
		b_k2_v2 = _mm512_set1_pd(B(k+1,2));
		b_k3_v2 = _mm512_set1_pd(B(k+1,3));
		b_k4_v2 = _mm512_set1_pd(B(k+1,4));
		b_k5_v2 = _mm512_set1_pd(B(k+1,5));
		b_k6_v2 = _mm512_set1_pd(B(k+1,6));
		b_k7_v2 = _mm512_set1_pd(B(k+1,7));
		c_00_10_20_30_40_50_60_70_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k0_v2, c_00_10_20_30_40_50_60_70_v);
		c_01_11_21_31_41_51_61_71_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k1_v2, c_01_11_21_31_41_51_61_71_v);
		c_02_12_22_32_42_52_62_72_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k2_v2, c_02_12_22_32_42_52_62_72_v);
		c_03_13_23_33_43_53_63_73_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k3_v2, c_03_13_23_33_43_53_63_73_v);
		c_04_14_24_34_44_54_64_74_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k4_v2, c_04_14_24_34_44_54_64_74_v);
		c_05_15_25_35_45_55_65_75_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k5_v2, c_05_15_25_35_45_55_65_75_v);
		c_06_16_26_36_46_56_66_76_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k6_v2, c_06_16_26_36_46_56_66_76_v);
		c_07_17_27_37_47_57_67_77_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k7_v2, c_07_17_27_37_47_57_67_77_v);
		
		a_0k_1k_2k_3k_4k_5k_6k_7k_v = _mm512_load_pd( (double *) &A(0,k+2));
		b_k0_v2 = _mm512_set1_pd(B(k+2,0));
		b_k1_v2 = _mm512_set1_pd(B(k+2,1));
		b_k2_v2 = _mm512_set1_pd(B(k+2,2));
		b_k3_v2 = _mm512_set1_pd(B(k+2,3));
		b_k4_v2 = _mm512_set1_pd(B(k+2,4));
		b_k5_v2 = _mm512_set1_pd(B(k+2,5));
		b_k6_v2 = _mm512_set1_pd(B(k+2,6));
		b_k7_v2 = _mm512_set1_pd(B(k+2,7));
		c_00_10_20_30_40_50_60_70_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k0_v2, c_00_10_20_30_40_50_60_70_v);
		c_01_11_21_31_41_51_61_71_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k1_v2, c_01_11_21_31_41_51_61_71_v);
		c_02_12_22_32_42_52_62_72_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k2_v2, c_02_12_22_32_42_52_62_72_v);
		c_03_13_23_33_43_53_63_73_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k3_v2, c_03_13_23_33_43_53_63_73_v);
		c_04_14_24_34_44_54_64_74_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k4_v2, c_04_14_24_34_44_54_64_74_v);
		c_05_15_25_35_45_55_65_75_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k5_v2, c_05_15_25_35_45_55_65_75_v);
		c_06_16_26_36_46_56_66_76_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k6_v2, c_06_16_26_36_46_56_66_76_v);
		c_07_17_27_37_47_57_67_77_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k7_v2, c_07_17_27_37_47_57_67_77_v);
		
		a_0k_1k_2k_3k_4k_5k_6k_7k_v = _mm512_load_pd( (double *) &A(0,k+3));
		b_k0_v2 = _mm512_set1_pd(B(k+3,0));
		b_k1_v2 = _mm512_set1_pd(B(k+3,1));
		b_k2_v2 = _mm512_set1_pd(B(k+3,2));
		b_k3_v2 = _mm512_set1_pd(B(k+3,3));
		b_k4_v2 = _mm512_set1_pd(B(k+3,4));
		b_k5_v2 = _mm512_set1_pd(B(k+3,5));
		b_k6_v2 = _mm512_set1_pd(B(k+3,6));
		b_k7_v2 = _mm512_set1_pd(B(k+3,7));
		c_00_10_20_30_40_50_60_70_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k0_v2, c_00_10_20_30_40_50_60_70_v);
		c_01_11_21_31_41_51_61_71_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k1_v2, c_01_11_21_31_41_51_61_71_v);
		c_02_12_22_32_42_52_62_72_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k2_v2, c_02_12_22_32_42_52_62_72_v);
		c_03_13_23_33_43_53_63_73_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k3_v2, c_03_13_23_33_43_53_63_73_v);
		c_04_14_24_34_44_54_64_74_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k4_v2, c_04_14_24_34_44_54_64_74_v);
		c_05_15_25_35_45_55_65_75_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k5_v2, c_05_15_25_35_45_55_65_75_v);
		c_06_16_26_36_46_56_66_76_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k6_v2, c_06_16_26_36_46_56_66_76_v);
		c_07_17_27_37_47_57_67_77_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k7_v2, c_07_17_27_37_47_57_67_77_v);

		
	}
	
	_mm512_store_pd(&C(0,0), c_00_10_20_30_40_50_60_70_v);
	_mm512_store_pd(&C(0,1), c_01_11_21_31_41_51_61_71_v);
	_mm512_store_pd(&C(0,2), c_02_12_22_32_42_52_62_72_v);
	_mm512_store_pd(&C(0,3), c_03_13_23_33_43_53_63_73_v);
	_mm512_store_pd(&C(0,4), c_04_14_24_34_44_54_64_74_v);
	_mm512_store_pd(&C(0,5), c_05_15_25_35_45_55_65_75_v);
	_mm512_store_pd(&C(0,6), c_06_16_26_36_46_56_66_76_v);
	_mm512_store_pd(&C(0,7), c_07_17_27_37_47_57_67_77_v);
  */
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void inner512_8x8(int K_, double* A, int lda, double* B, int ldb, double* C, int ldc) {
	
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
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k0_v2);
		//c_00_10_20_30_40_50_60_70_v = _mm512_add_pd(tmp_v, c_00_10_20_30_40_50_60_70_v);
		
		c_00_10_20_30_40_50_60_70_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k0_v2, c_00_10_20_30_40_50_60_70_v);
		c_01_11_21_31_41_51_61_71_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k1_v2, c_01_11_21_31_41_51_61_71_v);
		c_02_12_22_32_42_52_62_72_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k2_v2, c_02_12_22_32_42_52_62_72_v);
		c_03_13_23_33_43_53_63_73_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k3_v2, c_03_13_23_33_43_53_63_73_v);
		c_04_14_24_34_44_54_64_74_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k4_v2, c_04_14_24_34_44_54_64_74_v);
		c_05_15_25_35_45_55_65_75_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k5_v2, c_05_15_25_35_45_55_65_75_v);
		c_06_16_26_36_46_56_66_76_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k6_v2, c_06_16_26_36_46_56_66_76_v);
		c_07_17_27_37_47_57_67_77_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k7_v2, c_07_17_27_37_47_57_67_77_v);
		
		a_0k_1k_2k_3k_4k_5k_6k_7k_v = _mm512_load_pd( (double *) &A(0,k+1));
		b_k0_v2 = _mm512_set1_pd(B(k+1,0));
		b_k1_v2 = _mm512_set1_pd(B(k+1,1));
		b_k2_v2 = _mm512_set1_pd(B(k+1,2));
		b_k3_v2 = _mm512_set1_pd(B(k+1,3));
		b_k4_v2 = _mm512_set1_pd(B(k+1,4));
		b_k5_v2 = _mm512_set1_pd(B(k+1,5));
		b_k6_v2 = _mm512_set1_pd(B(k+1,6));
		b_k7_v2 = _mm512_set1_pd(B(k+1,7));
			c_00_10_20_30_40_50_60_70_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k0_v2, c_00_10_20_30_40_50_60_70_v);
		c_01_11_21_31_41_51_61_71_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k1_v2, c_01_11_21_31_41_51_61_71_v);
		c_02_12_22_32_42_52_62_72_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k2_v2, c_02_12_22_32_42_52_62_72_v);
		c_03_13_23_33_43_53_63_73_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k3_v2, c_03_13_23_33_43_53_63_73_v);
		c_04_14_24_34_44_54_64_74_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k4_v2, c_04_14_24_34_44_54_64_74_v);
		c_05_15_25_35_45_55_65_75_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k5_v2, c_05_15_25_35_45_55_65_75_v);
		c_06_16_26_36_46_56_66_76_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k6_v2, c_06_16_26_36_46_56_66_76_v);
		c_07_17_27_37_47_57_67_77_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k7_v2, c_07_17_27_37_47_57_67_77_v);
		
			a_0k_1k_2k_3k_4k_5k_6k_7k_v = _mm512_load_pd( (double *) &A(0,k+2));
		b_k0_v2 = _mm512_set1_pd(B(k+2,0));
		b_k1_v2 = _mm512_set1_pd(B(k+2,1));
		b_k2_v2 = _mm512_set1_pd(B(k+2,2));
		b_k3_v2 = _mm512_set1_pd(B(k+2,3));
		b_k4_v2 = _mm512_set1_pd(B(k+2,4));
		b_k5_v2 = _mm512_set1_pd(B(k+2,5));
		b_k6_v2 = _mm512_set1_pd(B(k+2,6));
		b_k7_v2 = _mm512_set1_pd(B(k+2,7));
			c_00_10_20_30_40_50_60_70_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k0_v2, c_00_10_20_30_40_50_60_70_v);
		c_01_11_21_31_41_51_61_71_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k1_v2, c_01_11_21_31_41_51_61_71_v);
		c_02_12_22_32_42_52_62_72_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k2_v2, c_02_12_22_32_42_52_62_72_v);
		c_03_13_23_33_43_53_63_73_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k3_v2, c_03_13_23_33_43_53_63_73_v);
		c_04_14_24_34_44_54_64_74_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k4_v2, c_04_14_24_34_44_54_64_74_v);
		c_05_15_25_35_45_55_65_75_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k5_v2, c_05_15_25_35_45_55_65_75_v);
		c_06_16_26_36_46_56_66_76_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k6_v2, c_06_16_26_36_46_56_66_76_v);
		c_07_17_27_37_47_57_67_77_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k7_v2, c_07_17_27_37_47_57_67_77_v);
		
			a_0k_1k_2k_3k_4k_5k_6k_7k_v = _mm512_load_pd( (double *) &A(0,k+3));
		b_k0_v2 = _mm512_set1_pd(B(k+3,0));
		b_k1_v2 = _mm512_set1_pd(B(k+3,1));
		b_k2_v2 = _mm512_set1_pd(B(k+3,2));
		b_k3_v2 = _mm512_set1_pd(B(k+3,3));
		b_k4_v2 = _mm512_set1_pd(B(k+3,4));
		b_k5_v2 = _mm512_set1_pd(B(k+3,5));
		b_k6_v2 = _mm512_set1_pd(B(k+3,6));
		b_k7_v2 = _mm512_set1_pd(B(k+3,7));
			c_00_10_20_30_40_50_60_70_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k0_v2, c_00_10_20_30_40_50_60_70_v);
		c_01_11_21_31_41_51_61_71_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k1_v2, c_01_11_21_31_41_51_61_71_v);
		c_02_12_22_32_42_52_62_72_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k2_v2, c_02_12_22_32_42_52_62_72_v);
		c_03_13_23_33_43_53_63_73_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k3_v2, c_03_13_23_33_43_53_63_73_v);
		c_04_14_24_34_44_54_64_74_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k4_v2, c_04_14_24_34_44_54_64_74_v);
		c_05_15_25_35_45_55_65_75_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k5_v2, c_05_15_25_35_45_55_65_75_v);
		c_06_16_26_36_46_56_66_76_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k6_v2, c_06_16_26_36_46_56_66_76_v);
		c_07_17_27_37_47_57_67_77_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k7_v2, c_07_17_27_37_47_57_67_77_v);
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k1_v2);
		//c_01_11_21_31_41_51_61_71_v = _mm512_add_pd(tmp_v, c_01_11_21_31_41_51_61_71_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k2_v2);
		//c_02_12_22_32_42_52_62_72_v = _mm512_add_pd(tmp_v, c_02_12_22_32_42_52_62_72_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k3_v2);
		//c_03_13_23_33_43_53_63_73_v = _mm512_add_pd(tmp_v, c_03_13_23_33_43_53_63_73_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k4_v2);
		//c_04_14_24_34_44_54_64_74_v = _mm512_add_pd(tmp_v, c_04_14_24_34_44_54_64_74_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k5_v2);
		//c_05_15_25_35_45_55_65_75_v = _mm512_add_pd(tmp_v, c_05_15_25_35_45_55_65_75_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k6_v2);
		//c_06_16_26_36_46_56_66_76_v = _mm512_add_pd(tmp_v, c_06_16_26_36_46_56_66_76_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k7_v2);
		//c_07_17_27_37_47_57_67_77_v = _mm512_add_pd(tmp_v, c_07_17_27_37_47_57_67_77_v);
		
		//a_0k_1k_2k_3k_4k_5k_6k_7k_v = _mm512_load_pd( (double *) &A(0,k+1));
		
		//b_k0_v2 = _mm512_set1_pd(B(k+1,0));
		//b_k1_v2 = _mm512_set1_pd(B(k+1,1));
		//b_k2_v2 = _mm512_set1_pd(B(k+1,2));
		//b_k3_v2 = _mm512_set1_pd(B(k+1,3));
		//b_k4_v2 = _mm512_set1_pd(B(k+1,4));
		//b_k5_v2 = _mm512_set1_pd(B(k+1,5));
		//b_k6_v2 = _mm512_set1_pd(B(k+1,6));
		//b_k7_v2 = _mm512_set1_pd(B(k+1,7));
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k0_v2);
		//c_00_10_20_30_40_50_60_70_v = _mm512_add_pd(tmp_v, c_00_10_20_30_40_50_60_70_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k1_v2);
		//c_01_11_21_31_41_51_61_71_v = _mm512_add_pd(tmp_v, c_01_11_21_31_41_51_61_71_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k2_v2);
		//c_02_12_22_32_42_52_62_72_v = _mm512_add_pd(tmp_v, c_02_12_22_32_42_52_62_72_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k3_v2);
		//c_03_13_23_33_43_53_63_73_v = _mm512_add_pd(tmp_v, c_03_13_23_33_43_53_63_73_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k4_v2);
		//c_04_14_24_34_44_54_64_74_v = _mm512_add_pd(tmp_v, c_04_14_24_34_44_54_64_74_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k5_v2);
		//c_05_15_25_35_45_55_65_75_v = _mm512_add_pd(tmp_v, c_05_15_25_35_45_55_65_75_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k6_v2);
		//c_06_16_26_36_46_56_66_76_v = _mm512_add_pd(tmp_v, c_06_16_26_36_46_56_66_76_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k7_v2);
		//c_07_17_27_37_47_57_67_77_v = _mm512_add_pd(tmp_v, c_07_17_27_37_47_57_67_77_v);
		
		//a_0k_1k_2k_3k_4k_5k_6k_7k_v = _mm512_load_pd( (double *) &A(0,k+2));
		
		//b_k0_v2 = _mm512_set1_pd(B(k+2,0));
		//b_k1_v2 = _mm512_set1_pd(B(k+2,1));
		//b_k2_v2 = _mm512_set1_pd(B(k+2,2));
		//b_k3_v2 = _mm512_set1_pd(B(k+2,3));
		//b_k4_v2 = _mm512_set1_pd(B(k+2,4));
		//b_k5_v2 = _mm512_set1_pd(B(k+2,5));
		//b_k6_v2 = _mm512_set1_pd(B(k+2,6));
		//b_k7_v2 = _mm512_set1_pd(B(k+2,7));
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k0_v2);
		//c_00_10_20_30_40_50_60_70_v = _mm512_add_pd(tmp_v, c_00_10_20_30_40_50_60_70_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k1_v2);
		//c_01_11_21_31_41_51_61_71_v = _mm512_add_pd(tmp_v, c_01_11_21_31_41_51_61_71_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k2_v2);
		//c_02_12_22_32_42_52_62_72_v = _mm512_add_pd(tmp_v, c_02_12_22_32_42_52_62_72_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k3_v2);
		//c_03_13_23_33_43_53_63_73_v = _mm512_add_pd(tmp_v, c_03_13_23_33_43_53_63_73_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k4_v2);
		//c_04_14_24_34_44_54_64_74_v = _mm512_add_pd(tmp_v, c_04_14_24_34_44_54_64_74_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k5_v2);
		//c_05_15_25_35_45_55_65_75_v = _mm512_add_pd(tmp_v, c_05_15_25_35_45_55_65_75_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k6_v2);
		//c_06_16_26_36_46_56_66_76_v = _mm512_add_pd(tmp_v, c_06_16_26_36_46_56_66_76_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k7_v2);
		//c_07_17_27_37_47_57_67_77_v = _mm512_add_pd(tmp_v, c_07_17_27_37_47_57_67_77_v);
		
		//a_0k_1k_2k_3k_4k_5k_6k_7k_v = _mm512_load_pd( (double *) &A(0,k+3));
		
		//b_k0_v2 = _mm512_set1_pd(B(k+3,0));
		//b_k1_v2 = _mm512_set1_pd(B(k+3,1));
		//b_k2_v2 = _mm512_set1_pd(B(k+3,2));
		//b_k3_v2 = _mm512_set1_pd(B(k+3,3));
		//b_k4_v2 = _mm512_set1_pd(B(k+3,4));
		//b_k5_v2 = _mm512_set1_pd(B(k+3,5));
		//b_k6_v2 = _mm512_set1_pd(B(k+3,6));
		//b_k7_v2 = _mm512_set1_pd(B(k+3,7));
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k0_v2);
		//c_00_10_20_30_40_50_60_70_v = _mm512_add_pd(tmp_v, c_00_10_20_30_40_50_60_70_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k1_v2);
		//c_01_11_21_31_41_51_61_71_v = _mm512_add_pd(tmp_v, c_01_11_21_31_41_51_61_71_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k2_v2);
		//c_02_12_22_32_42_52_62_72_v = _mm512_add_pd(tmp_v, c_02_12_22_32_42_52_62_72_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k3_v2);
		//c_03_13_23_33_43_53_63_73_v = _mm512_add_pd(tmp_v, c_03_13_23_33_43_53_63_73_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k4_v2);
		//c_04_14_24_34_44_54_64_74_v = _mm512_add_pd(tmp_v, c_04_14_24_34_44_54_64_74_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k5_v2);
		//c_05_15_25_35_45_55_65_75_v = _mm512_add_pd(tmp_v, c_05_15_25_35_45_55_65_75_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k6_v2);
		//c_06_16_26_36_46_56_66_76_v = _mm512_add_pd(tmp_v, c_06_16_26_36_46_56_66_76_v);
		
		//tmp_v = a_0k_1k_2k_3k_4k_5k_6k_7k_v;
		//tmp_v = _mm512_mul_pd(tmp_v, b_k7_v2);
		//c_07_17_27_37_47_57_67_77_v = _mm512_add_pd(tmp_v, c_07_17_27_37_47_57_67_77_v);
		
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


// pack A in contiguous memory
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
	
	//printf("a"); // only works with printf
}

// for blocking
void innerKernel(int M_, int N_, int K_, double* A, int lda, double* B, int ldb, double* C, int ldc, int first) {
	
	//double
	//packedA[M_*K_];
	
	//double packedB[kc * nb];


	for (int n = 0; n < N_; n+=8) {
		//if (first) packB(K_, &B(0,n), ldb, &packedB[n*K_]);
		for (int m = 0; m < M_; m+=8) {
		  //if (n == 0) packA(K_, &A(m,0), lda, &packedA[m*K_]);
		  //inner512_8x8(K_, &packedA[m*K_], 8, &B(0,n),ldb ,&C(m,n), ldc);	  
		  
		  //inner512_8x8(K_, &A(m,0), lda, &packedB[n*K_],K_ ,&C(m,n), ldc);
		  
		  inner512_8x8(K_, &A(m,0), lda, &B(0,n),ldb ,&C(m,n), ldc);
		}
	}
}

void dgemm_opt(double* A, double* B, double* C) {
  
 //for (int n = 0; n < N; n+=4) {
    //for (int m = 0; m < M; m+=4) {
		
	  ////inner4(&A[m], &B[n*K], &C[n*M + m]);
		
	  ////innerNew(&A(m,0), &B(0,n), &C(m,n));
	  //inner256(&A(m,0), &B(0,n), &C(m,n));
    //}
  //}
  
  
   //for (int n = 0; n < N; n+=8) {
    //for (int m = 0; m < M; m+=8) {
		
	  ////inner4(&A[m], &B[n*K], &C[n*M + m]);
		
	  ////innerNew(&A(m,0), &B(0,n), &C(m,n));
	  ////inner512(&A(m,0), &B(0,n), &C(m,n));
	  //inner512_8x8(&A(m,0), &B(0,n), &C(m,n));
    //}
  //}
  int S = 128;
  int kb, mb;
  int lda = S;
  int ldb = S;
  int ldc = S;
  
  for (int k = 0; k < S; k+=K) {
	  kb = std::min(S-k, K);
	  for (int m = 0; m < S; m+=MC) {
		  mb = std::min(S-m, MC);
		  innerKernel(mb, S, kb, &A(m,k), lda, &B(k,0), ldb, &C(m,0), ldc, m==0);
	  } 
	 }
  
  
  
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * A: M x K, ld M
 * B: K x N, ld K
 * C: M x N, ld S
 */
// arch independent and slow, but works
void microkernel_gen2(const double* A, const double* B, double* C, int S) {
#pragma message ("LIBXSMM KERNEL COMPILATION WARNING: compiling arch-independent gemm kernel in: " __FILE__)
  unsigned int l_m = 0;
  unsigned int l_n = 0;
  unsigned int l_k = 0;

  for ( l_n = 0; l_n < N; l_n++ ) {
    for ( l_k = 0; l_k < K; l_k++ ) {
      #pragma simd
      for ( l_m = 0; l_m < M; l_m++ ) {
        C[(l_n*S)+l_m] += A[(l_k*M)+l_m] * B[(l_n*K)+l_k];
      }
    }
  }
}



// everything set to 128, works for square, no packing
void microkernel_gen(const double* A, const double* B, double* C) {
#ifdef __AVX512F__
  __asm__ __volatile__("movq %0, %%rdi\n\t"
                       "movq %1, %%rsi\n\t"
                       "movq %2, %%rdx\n\t"
                       "movq $0, %%r12\n\t"
                       "movq $0, %%r13\n\t"
                       "movq $0, %%r14\n\t"
                       "0:\n\t"
                       "addq $26, %%r13\n\t"
                       "movq $0, %%r12\n\t"
                       "1:\n\t"
                       "addq $8, %%r12\n\t"
                       "vmovupd 0(%%rdx), %%zmm6\n\t"
                       "vmovupd 1024(%%rdx), %%zmm7\n\t"
                       "vmovupd 2048(%%rdx), %%zmm8\n\t"
                       "vmovupd 3072(%%rdx), %%zmm9\n\t"
                       "vmovupd 4096(%%rdx), %%zmm10\n\t"
                       "vmovupd 5120(%%rdx), %%zmm11\n\t"
                       "vmovupd 6144(%%rdx), %%zmm12\n\t"
                       "vmovupd 7168(%%rdx), %%zmm13\n\t"
                       "vmovupd 8192(%%rdx), %%zmm14\n\t"
                       "vmovupd 9216(%%rdx), %%zmm15\n\t"
                       "vmovupd 10240(%%rdx), %%zmm16\n\t"
                       "vmovupd 11264(%%rdx), %%zmm17\n\t"
                       "vmovupd 12288(%%rdx), %%zmm18\n\t"
                       "vmovupd 13312(%%rdx), %%zmm19\n\t"
                       "vmovupd 14336(%%rdx), %%zmm20\n\t"
                       "vmovupd 15360(%%rdx), %%zmm21\n\t"
                       "vmovupd 16384(%%rdx), %%zmm22\n\t"
                       "vmovupd 17408(%%rdx), %%zmm23\n\t"
                       "vmovupd 18432(%%rdx), %%zmm24\n\t"
                       "vmovupd 19456(%%rdx), %%zmm25\n\t"
                       "vmovupd 20480(%%rdx), %%zmm26\n\t"
                       "vmovupd 21504(%%rdx), %%zmm27\n\t"
                       "vmovupd 22528(%%rdx), %%zmm28\n\t"
                       "vmovupd 23552(%%rdx), %%zmm29\n\t"
                       "vmovupd 24576(%%rdx), %%zmm30\n\t"
                       "vmovupd 25600(%%rdx), %%zmm31\n\t"
                       "pushq %%rdx\n\t"
                       "movq $0, %%r14\n\t"
                       "2:\n\t"
                       "addq $8, %%r14\n\t"
                       "movq $1024, %%r15\n\t"
                       "movq $3072, %%rax\n\t"
                       "movq $5120, %%rbx\n\t"
                       "movq $7168, %%r11\n\t"
                       "movq %%rsi, %%r10\n\t"
                       "addq $9216, %%r10\n\t"
                       "movq %%rsi, %%rdx\n\t"
                       "addq $18432, %%rdx\n\t"
                       "vmovupd 0(%%rdi), %%zmm0\n\t"
                       "vmovupd 1024(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 0(%%rsi)%{1to8%}, %%zmm0, %%zmm6\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm7\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm8\n\t"
                       "vfmadd231pd 0(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm9\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm10\n\t"
                       "vfmadd231pd 0(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm11\n\t"
                       "vfmadd231pd 0(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm12\n\t"
                       "vfmadd231pd 0(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm13\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm14\n\t"
                       "vfmadd231pd 0(%%r10)%{1to8%}, %%zmm0, %%zmm15\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 0(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 0(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 0(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 0(%%r10,%%r11,1)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,8)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 0(%%rdx)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 0(%%rdx,%%r15,1)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 0(%%rdx,%%r15,2)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 0(%%rdx,%%rax,1)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 0(%%rdx,%%r15,4)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 0(%%rdx,%%rbx,1)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 0(%%rdx,%%rax,2)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 0(%%rdx,%%r11,1)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovupd 2048(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 8(%%rsi)%{1to8%}, %%zmm1, %%zmm6\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm7\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm8\n\t"
                       "vfmadd231pd 8(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm9\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm10\n\t"
                       "vfmadd231pd 8(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm11\n\t"
                       "vfmadd231pd 8(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm12\n\t"
                       "vfmadd231pd 8(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm13\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm14\n\t"
                       "vfmadd231pd 8(%%r10)%{1to8%}, %%zmm1, %%zmm15\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 8(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 8(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 8(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 8(%%r10,%%r11,1)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,8)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 8(%%rdx)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 8(%%rdx,%%r15,1)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 8(%%rdx,%%r15,2)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 8(%%rdx,%%rax,1)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 8(%%rdx,%%r15,4)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 8(%%rdx,%%rbx,1)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 8(%%rdx,%%rax,2)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 8(%%rdx,%%r11,1)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "vmovupd 3072(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 16(%%rsi)%{1to8%}, %%zmm0, %%zmm6\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm7\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm8\n\t"
                       "vfmadd231pd 16(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm9\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm10\n\t"
                       "vfmadd231pd 16(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm11\n\t"
                       "vfmadd231pd 16(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm12\n\t"
                       "vfmadd231pd 16(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm13\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm14\n\t"
                       "vfmadd231pd 16(%%r10)%{1to8%}, %%zmm0, %%zmm15\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 16(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 16(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 16(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 16(%%r10,%%r11,1)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,8)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 16(%%rdx)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 16(%%rdx,%%r15,1)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 16(%%rdx,%%r15,2)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 16(%%rdx,%%rax,1)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 16(%%rdx,%%r15,4)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 16(%%rdx,%%rbx,1)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 16(%%rdx,%%rax,2)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 16(%%rdx,%%r11,1)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovupd 4096(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 24(%%rsi)%{1to8%}, %%zmm1, %%zmm6\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm7\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm8\n\t"
                       "vfmadd231pd 24(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm9\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm10\n\t"
                       "vfmadd231pd 24(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm11\n\t"
                       "vfmadd231pd 24(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm12\n\t"
                       "vfmadd231pd 24(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm13\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm14\n\t"
                       "vfmadd231pd 24(%%r10)%{1to8%}, %%zmm1, %%zmm15\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 24(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 24(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 24(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 24(%%r10,%%r11,1)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,8)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 24(%%rdx)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 24(%%rdx,%%r15,1)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 24(%%rdx,%%r15,2)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 24(%%rdx,%%rax,1)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 24(%%rdx,%%r15,4)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 24(%%rdx,%%rbx,1)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 24(%%rdx,%%rax,2)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 24(%%rdx,%%r11,1)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "vmovupd 5120(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 32(%%rsi)%{1to8%}, %%zmm0, %%zmm6\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm7\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm8\n\t"
                       "vfmadd231pd 32(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm9\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm10\n\t"
                       "vfmadd231pd 32(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm11\n\t"
                       "vfmadd231pd 32(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm12\n\t"
                       "vfmadd231pd 32(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm13\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm14\n\t"
                       "vfmadd231pd 32(%%r10)%{1to8%}, %%zmm0, %%zmm15\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 32(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 32(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 32(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 32(%%r10,%%r11,1)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,8)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 32(%%rdx)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 32(%%rdx,%%r15,1)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 32(%%rdx,%%r15,2)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 32(%%rdx,%%rax,1)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 32(%%rdx,%%r15,4)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 32(%%rdx,%%rbx,1)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 32(%%rdx,%%rax,2)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 32(%%rdx,%%r11,1)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovupd 6144(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 40(%%rsi)%{1to8%}, %%zmm1, %%zmm6\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm7\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm8\n\t"
                       "vfmadd231pd 40(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm9\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm10\n\t"
                       "vfmadd231pd 40(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm11\n\t"
                       "vfmadd231pd 40(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm12\n\t"
                       "vfmadd231pd 40(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm13\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm14\n\t"
                       "vfmadd231pd 40(%%r10)%{1to8%}, %%zmm1, %%zmm15\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 40(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 40(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 40(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 40(%%r10,%%r11,1)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,8)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 40(%%rdx)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 40(%%rdx,%%r15,1)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 40(%%rdx,%%r15,2)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 40(%%rdx,%%rax,1)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 40(%%rdx,%%r15,4)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 40(%%rdx,%%rbx,1)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 40(%%rdx,%%rax,2)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 40(%%rdx,%%r11,1)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "vmovupd 7168(%%rdi), %%zmm1\n\t"
                       "addq $8192, %%rdi\n\t"
                       "vfmadd231pd 48(%%rsi)%{1to8%}, %%zmm0, %%zmm6\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm7\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm8\n\t"
                       "vfmadd231pd 48(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm9\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm10\n\t"
                       "vfmadd231pd 48(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm11\n\t"
                       "vfmadd231pd 48(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm12\n\t"
                       "vfmadd231pd 48(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm13\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm14\n\t"
                       "vfmadd231pd 48(%%r10)%{1to8%}, %%zmm0, %%zmm15\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 48(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 48(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 48(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 48(%%r10,%%r11,1)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,8)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 48(%%rdx)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 48(%%rdx,%%r15,1)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 48(%%rdx,%%r15,2)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 48(%%rdx,%%rax,1)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 48(%%rdx,%%r15,4)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 48(%%rdx,%%rbx,1)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 48(%%rdx,%%rax,2)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 48(%%rdx,%%r11,1)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "addq $8192, %%rdi\n\t"
                       "vfmadd231pd 56(%%rsi)%{1to8%}, %%zmm1, %%zmm6\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm7\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm8\n\t"
                       "vfmadd231pd 56(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm9\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm10\n\t"
                       "vfmadd231pd 56(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm11\n\t"
                       "vfmadd231pd 56(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm12\n\t"
                       "vfmadd231pd 56(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm13\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm14\n\t"
                       "vfmadd231pd 56(%%r10)%{1to8%}, %%zmm1, %%zmm15\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 56(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 56(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 56(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 56(%%r10,%%r11,1)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,8)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 56(%%rdx)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 56(%%rdx,%%r15,1)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 56(%%rdx,%%r15,2)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 56(%%rdx,%%rax,1)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 56(%%rdx,%%r15,4)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 56(%%rdx,%%rbx,1)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 56(%%rdx,%%rax,2)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 56(%%rdx,%%r11,1)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "subq $8192, %%rdi\n\t"
                       "addq $64, %%rsi\n\t"
                       "cmpq $128, %%r14\n\t"
                       "jl 2b\n\t"
                       "subq $1024, %%rsi\n\t"
                       "popq %%rdx\n\t"
                       "vmovupd %%zmm6, 0(%%rdx)\n\t"
                       "vmovupd %%zmm7, 1024(%%rdx)\n\t"
                       "vmovupd %%zmm8, 2048(%%rdx)\n\t"
                       "vmovupd %%zmm9, 3072(%%rdx)\n\t"
                       "vmovupd %%zmm10, 4096(%%rdx)\n\t"
                       "vmovupd %%zmm11, 5120(%%rdx)\n\t"
                       "vmovupd %%zmm12, 6144(%%rdx)\n\t"
                       "vmovupd %%zmm13, 7168(%%rdx)\n\t"
                       "vmovupd %%zmm14, 8192(%%rdx)\n\t"
                       "vmovupd %%zmm15, 9216(%%rdx)\n\t"
                       "vmovupd %%zmm16, 10240(%%rdx)\n\t"
                       "vmovupd %%zmm17, 11264(%%rdx)\n\t"
                       "vmovupd %%zmm18, 12288(%%rdx)\n\t"
                       "vmovupd %%zmm19, 13312(%%rdx)\n\t"
                       "vmovupd %%zmm20, 14336(%%rdx)\n\t"
                       "vmovupd %%zmm21, 15360(%%rdx)\n\t"
                       "vmovupd %%zmm22, 16384(%%rdx)\n\t"
                       "vmovupd %%zmm23, 17408(%%rdx)\n\t"
                       "vmovupd %%zmm24, 18432(%%rdx)\n\t"
                       "vmovupd %%zmm25, 19456(%%rdx)\n\t"
                       "vmovupd %%zmm26, 20480(%%rdx)\n\t"
                       "vmovupd %%zmm27, 21504(%%rdx)\n\t"
                       "vmovupd %%zmm28, 22528(%%rdx)\n\t"
                       "vmovupd %%zmm29, 23552(%%rdx)\n\t"
                       "vmovupd %%zmm30, 24576(%%rdx)\n\t"
                       "vmovupd %%zmm31, 25600(%%rdx)\n\t"
                       "addq $64, %%rdx\n\t"
                       "subq $131008, %%rdi\n\t"
                       "cmpq $128, %%r12\n\t"
                       "jl 1b\n\t"
                       "addq $25600, %%rdx\n\t"
                       "addq $26624, %%rsi\n\t"
                       "subq $1024, %%rdi\n\t"
                       "cmpq $78, %%r13\n\t"
                       "jl 0b\n\t"
                       "0:\n\t"
                       "addq $25, %%r13\n\t"
                       "movq $0, %%r12\n\t"
                       "1:\n\t"
                       "addq $8, %%r12\n\t"
                       "vmovupd 0(%%rdx), %%zmm7\n\t"
                       "vmovupd 1024(%%rdx), %%zmm8\n\t"
                       "vmovupd 2048(%%rdx), %%zmm9\n\t"
                       "vmovupd 3072(%%rdx), %%zmm10\n\t"
                       "vmovupd 4096(%%rdx), %%zmm11\n\t"
                       "vmovupd 5120(%%rdx), %%zmm12\n\t"
                       "vmovupd 6144(%%rdx), %%zmm13\n\t"
                       "vmovupd 7168(%%rdx), %%zmm14\n\t"
                       "vmovupd 8192(%%rdx), %%zmm15\n\t"
                       "vmovupd 9216(%%rdx), %%zmm16\n\t"
                       "vmovupd 10240(%%rdx), %%zmm17\n\t"
                       "vmovupd 11264(%%rdx), %%zmm18\n\t"
                       "vmovupd 12288(%%rdx), %%zmm19\n\t"
                       "vmovupd 13312(%%rdx), %%zmm20\n\t"
                       "vmovupd 14336(%%rdx), %%zmm21\n\t"
                       "vmovupd 15360(%%rdx), %%zmm22\n\t"
                       "vmovupd 16384(%%rdx), %%zmm23\n\t"
                       "vmovupd 17408(%%rdx), %%zmm24\n\t"
                       "vmovupd 18432(%%rdx), %%zmm25\n\t"
                       "vmovupd 19456(%%rdx), %%zmm26\n\t"
                       "vmovupd 20480(%%rdx), %%zmm27\n\t"
                       "vmovupd 21504(%%rdx), %%zmm28\n\t"
                       "vmovupd 22528(%%rdx), %%zmm29\n\t"
                       "vmovupd 23552(%%rdx), %%zmm30\n\t"
                       "vmovupd 24576(%%rdx), %%zmm31\n\t"
                       "pushq %%rdx\n\t"
                       "movq $0, %%r14\n\t"
                       "2:\n\t"
                       "addq $8, %%r14\n\t"
                       "movq $1024, %%r15\n\t"
                       "movq $3072, %%rax\n\t"
                       "movq $5120, %%rbx\n\t"
                       "movq $7168, %%r11\n\t"
                       "movq %%rsi, %%r10\n\t"
                       "addq $9216, %%r10\n\t"
                       "movq %%rsi, %%rdx\n\t"
                       "addq $18432, %%rdx\n\t"
                       "vmovupd 0(%%rdi), %%zmm0\n\t"
                       "vmovupd 1024(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 0(%%rsi)%{1to8%}, %%zmm0, %%zmm7\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm8\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm9\n\t"
                       "vfmadd231pd 0(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm10\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm11\n\t"
                       "vfmadd231pd 0(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm12\n\t"
                       "vfmadd231pd 0(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm13\n\t"
                       "vfmadd231pd 0(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm14\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm15\n\t"
                       "vfmadd231pd 0(%%r10)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 0(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 0(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 0(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 0(%%r10,%%r11,1)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,8)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 0(%%rdx)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 0(%%rdx,%%r15,1)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 0(%%rdx,%%r15,2)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 0(%%rdx,%%rax,1)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 0(%%rdx,%%r15,4)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 0(%%rdx,%%rbx,1)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 0(%%rdx,%%rax,2)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovupd 2048(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 8(%%rsi)%{1to8%}, %%zmm1, %%zmm7\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm8\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm9\n\t"
                       "vfmadd231pd 8(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm10\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm11\n\t"
                       "vfmadd231pd 8(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm12\n\t"
                       "vfmadd231pd 8(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm13\n\t"
                       "vfmadd231pd 8(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm14\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm15\n\t"
                       "vfmadd231pd 8(%%r10)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 8(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 8(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 8(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 8(%%r10,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,8)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 8(%%rdx)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 8(%%rdx,%%r15,1)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 8(%%rdx,%%r15,2)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 8(%%rdx,%%rax,1)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 8(%%rdx,%%r15,4)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 8(%%rdx,%%rbx,1)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 8(%%rdx,%%rax,2)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "vmovupd 3072(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 16(%%rsi)%{1to8%}, %%zmm0, %%zmm7\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm8\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm9\n\t"
                       "vfmadd231pd 16(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm10\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm11\n\t"
                       "vfmadd231pd 16(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm12\n\t"
                       "vfmadd231pd 16(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm13\n\t"
                       "vfmadd231pd 16(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm14\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm15\n\t"
                       "vfmadd231pd 16(%%r10)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 16(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 16(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 16(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 16(%%r10,%%r11,1)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,8)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 16(%%rdx)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 16(%%rdx,%%r15,1)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 16(%%rdx,%%r15,2)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 16(%%rdx,%%rax,1)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 16(%%rdx,%%r15,4)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 16(%%rdx,%%rbx,1)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 16(%%rdx,%%rax,2)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovupd 4096(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 24(%%rsi)%{1to8%}, %%zmm1, %%zmm7\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm8\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm9\n\t"
                       "vfmadd231pd 24(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm10\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm11\n\t"
                       "vfmadd231pd 24(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm12\n\t"
                       "vfmadd231pd 24(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm13\n\t"
                       "vfmadd231pd 24(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm14\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm15\n\t"
                       "vfmadd231pd 24(%%r10)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 24(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 24(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 24(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 24(%%r10,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,8)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 24(%%rdx)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 24(%%rdx,%%r15,1)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 24(%%rdx,%%r15,2)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 24(%%rdx,%%rax,1)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 24(%%rdx,%%r15,4)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 24(%%rdx,%%rbx,1)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 24(%%rdx,%%rax,2)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "vmovupd 5120(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 32(%%rsi)%{1to8%}, %%zmm0, %%zmm7\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm8\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm9\n\t"
                       "vfmadd231pd 32(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm10\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm11\n\t"
                       "vfmadd231pd 32(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm12\n\t"
                       "vfmadd231pd 32(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm13\n\t"
                       "vfmadd231pd 32(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm14\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm15\n\t"
                       "vfmadd231pd 32(%%r10)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 32(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 32(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 32(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 32(%%r10,%%r11,1)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,8)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 32(%%rdx)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 32(%%rdx,%%r15,1)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 32(%%rdx,%%r15,2)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 32(%%rdx,%%rax,1)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 32(%%rdx,%%r15,4)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 32(%%rdx,%%rbx,1)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 32(%%rdx,%%rax,2)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovupd 6144(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 40(%%rsi)%{1to8%}, %%zmm1, %%zmm7\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm8\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm9\n\t"
                       "vfmadd231pd 40(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm10\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm11\n\t"
                       "vfmadd231pd 40(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm12\n\t"
                       "vfmadd231pd 40(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm13\n\t"
                       "vfmadd231pd 40(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm14\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm15\n\t"
                       "vfmadd231pd 40(%%r10)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 40(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 40(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 40(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 40(%%r10,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,8)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 40(%%rdx)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 40(%%rdx,%%r15,1)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 40(%%rdx,%%r15,2)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 40(%%rdx,%%rax,1)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 40(%%rdx,%%r15,4)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 40(%%rdx,%%rbx,1)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 40(%%rdx,%%rax,2)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "vmovupd 7168(%%rdi), %%zmm1\n\t"
                       "addq $8192, %%rdi\n\t"
                       "vfmadd231pd 48(%%rsi)%{1to8%}, %%zmm0, %%zmm7\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm8\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm9\n\t"
                       "vfmadd231pd 48(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm10\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm11\n\t"
                       "vfmadd231pd 48(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm12\n\t"
                       "vfmadd231pd 48(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm13\n\t"
                       "vfmadd231pd 48(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm14\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm15\n\t"
                       "vfmadd231pd 48(%%r10)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 48(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 48(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 48(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 48(%%r10,%%r11,1)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,8)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 48(%%rdx)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 48(%%rdx,%%r15,1)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 48(%%rdx,%%r15,2)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 48(%%rdx,%%rax,1)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 48(%%rdx,%%r15,4)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 48(%%rdx,%%rbx,1)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 48(%%rdx,%%rax,2)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "addq $8192, %%rdi\n\t"
                       "vfmadd231pd 56(%%rsi)%{1to8%}, %%zmm1, %%zmm7\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm8\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm9\n\t"
                       "vfmadd231pd 56(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm10\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm11\n\t"
                       "vfmadd231pd 56(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm12\n\t"
                       "vfmadd231pd 56(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm13\n\t"
                       "vfmadd231pd 56(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm14\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm15\n\t"
                       "vfmadd231pd 56(%%r10)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 56(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 56(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 56(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 56(%%r10,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,8)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 56(%%rdx)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 56(%%rdx,%%r15,1)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 56(%%rdx,%%r15,2)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 56(%%rdx,%%rax,1)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 56(%%rdx,%%r15,4)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 56(%%rdx,%%rbx,1)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 56(%%rdx,%%rax,2)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "subq $8192, %%rdi\n\t"
                       "addq $64, %%rsi\n\t"
                       "cmpq $128, %%r14\n\t"
                       "jl 2b\n\t"
                       "subq $1024, %%rsi\n\t"
                       "popq %%rdx\n\t"
                       "vmovupd %%zmm7, 0(%%rdx)\n\t"
                       "vmovupd %%zmm8, 1024(%%rdx)\n\t"
                       "vmovupd %%zmm9, 2048(%%rdx)\n\t"
                       "vmovupd %%zmm10, 3072(%%rdx)\n\t"
                       "vmovupd %%zmm11, 4096(%%rdx)\n\t"
                       "vmovupd %%zmm12, 5120(%%rdx)\n\t"
                       "vmovupd %%zmm13, 6144(%%rdx)\n\t"
                       "vmovupd %%zmm14, 7168(%%rdx)\n\t"
                       "vmovupd %%zmm15, 8192(%%rdx)\n\t"
                       "vmovupd %%zmm16, 9216(%%rdx)\n\t"
                       "vmovupd %%zmm17, 10240(%%rdx)\n\t"
                       "vmovupd %%zmm18, 11264(%%rdx)\n\t"
                       "vmovupd %%zmm19, 12288(%%rdx)\n\t"
                       "vmovupd %%zmm20, 13312(%%rdx)\n\t"
                       "vmovupd %%zmm21, 14336(%%rdx)\n\t"
                       "vmovupd %%zmm22, 15360(%%rdx)\n\t"
                       "vmovupd %%zmm23, 16384(%%rdx)\n\t"
                       "vmovupd %%zmm24, 17408(%%rdx)\n\t"
                       "vmovupd %%zmm25, 18432(%%rdx)\n\t"
                       "vmovupd %%zmm26, 19456(%%rdx)\n\t"
                       "vmovupd %%zmm27, 20480(%%rdx)\n\t"
                       "vmovupd %%zmm28, 21504(%%rdx)\n\t"
                       "vmovupd %%zmm29, 22528(%%rdx)\n\t"
                       "vmovupd %%zmm30, 23552(%%rdx)\n\t"
                       "vmovupd %%zmm31, 24576(%%rdx)\n\t"
                       "addq $64, %%rdx\n\t"
                       "subq $131008, %%rdi\n\t"
                       "cmpq $128, %%r12\n\t"
                       "jl 1b\n\t"
                       "addq $24576, %%rdx\n\t"
                       "addq $25600, %%rsi\n\t"
                       "subq $1024, %%rdi\n\t"
                       "cmpq $128, %%r13\n\t"
                       "jl 0b\n\t"
                       : : "m"(A), "m"(B), "m"(C) : "k1","rax","rbx","rcx","rdx","rdi","rsi","r8","r9","r10","r11","r12","r13","r14","r15","zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15","zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");
#else
#pragma message ("LIBXSMM KERNEL COMPILATION ERROR in: " __FILE__)
#error No kernel was compiled, lacking support for current architecture?
#endif
}

// S 256, everything else 128
void microkernel_gen_256(const double* A, const double* B, double* C) {
#ifdef __AVX512F__
  __asm__ __volatile__("movq %0, %%rdi\n\t"
                       "movq %1, %%rsi\n\t"
                       "movq %2, %%rdx\n\t"
                       "movq $0, %%r12\n\t"
                       "movq $0, %%r13\n\t"
                       "movq $0, %%r14\n\t"
                       "0:\n\t"
                       "addq $26, %%r13\n\t"
                       "movq $0, %%r12\n\t"
                       "1:\n\t"
                       "addq $8, %%r12\n\t"
                       "vmovupd 0(%%rdx), %%zmm6\n\t"
                       "vmovupd 2048(%%rdx), %%zmm7\n\t"
                       "vmovupd 4096(%%rdx), %%zmm8\n\t"
                       "vmovupd 6144(%%rdx), %%zmm9\n\t"
                       "vmovupd 8192(%%rdx), %%zmm10\n\t"
                       "vmovupd 10240(%%rdx), %%zmm11\n\t"
                       "vmovupd 12288(%%rdx), %%zmm12\n\t"
                       "vmovupd 14336(%%rdx), %%zmm13\n\t"
                       "vmovupd 16384(%%rdx), %%zmm14\n\t"
                       "vmovupd 18432(%%rdx), %%zmm15\n\t"
                       "vmovupd 20480(%%rdx), %%zmm16\n\t"
                       "vmovupd 22528(%%rdx), %%zmm17\n\t"
                       "vmovupd 24576(%%rdx), %%zmm18\n\t"
                       "vmovupd 26624(%%rdx), %%zmm19\n\t"
                       "vmovupd 28672(%%rdx), %%zmm20\n\t"
                       "vmovupd 30720(%%rdx), %%zmm21\n\t"
                       "vmovupd 32768(%%rdx), %%zmm22\n\t"
                       "vmovupd 34816(%%rdx), %%zmm23\n\t"
                       "vmovupd 36864(%%rdx), %%zmm24\n\t"
                       "vmovupd 38912(%%rdx), %%zmm25\n\t"
                       "vmovupd 40960(%%rdx), %%zmm26\n\t"
                       "vmovupd 43008(%%rdx), %%zmm27\n\t"
                       "vmovupd 45056(%%rdx), %%zmm28\n\t"
                       "vmovupd 47104(%%rdx), %%zmm29\n\t"
                       "vmovupd 49152(%%rdx), %%zmm30\n\t"
                       "vmovupd 51200(%%rdx), %%zmm31\n\t"
                       "pushq %%rdx\n\t"
                       "movq $0, %%r14\n\t"
                       "2:\n\t"
                       "addq $8, %%r14\n\t"
                       "movq $1024, %%r15\n\t"
                       "movq $3072, %%rax\n\t"
                       "movq $5120, %%rbx\n\t"
                       "movq $7168, %%r11\n\t"
                       "movq %%rsi, %%r10\n\t"
                       "addq $9216, %%r10\n\t"
                       "movq %%rsi, %%rdx\n\t"
                       "addq $18432, %%rdx\n\t"
                       "vmovupd 0(%%rdi), %%zmm0\n\t"
                       "vmovupd 1024(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 0(%%rsi)%{1to8%}, %%zmm0, %%zmm6\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm7\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm8\n\t"
                       "vfmadd231pd 0(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm9\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm10\n\t"
                       "vfmadd231pd 0(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm11\n\t"
                       "vfmadd231pd 0(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm12\n\t"
                       "vfmadd231pd 0(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm13\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm14\n\t"
                       "vfmadd231pd 0(%%r10)%{1to8%}, %%zmm0, %%zmm15\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 0(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 0(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 0(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 0(%%r10,%%r11,1)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,8)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 0(%%rdx)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 0(%%rdx,%%r15,1)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 0(%%rdx,%%r15,2)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 0(%%rdx,%%rax,1)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 0(%%rdx,%%r15,4)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 0(%%rdx,%%rbx,1)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 0(%%rdx,%%rax,2)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 0(%%rdx,%%r11,1)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovupd 2048(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 8(%%rsi)%{1to8%}, %%zmm1, %%zmm6\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm7\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm8\n\t"
                       "vfmadd231pd 8(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm9\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm10\n\t"
                       "vfmadd231pd 8(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm11\n\t"
                       "vfmadd231pd 8(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm12\n\t"
                       "vfmadd231pd 8(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm13\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm14\n\t"
                       "vfmadd231pd 8(%%r10)%{1to8%}, %%zmm1, %%zmm15\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 8(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 8(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 8(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 8(%%r10,%%r11,1)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,8)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 8(%%rdx)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 8(%%rdx,%%r15,1)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 8(%%rdx,%%r15,2)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 8(%%rdx,%%rax,1)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 8(%%rdx,%%r15,4)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 8(%%rdx,%%rbx,1)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 8(%%rdx,%%rax,2)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 8(%%rdx,%%r11,1)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "vmovupd 3072(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 16(%%rsi)%{1to8%}, %%zmm0, %%zmm6\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm7\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm8\n\t"
                       "vfmadd231pd 16(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm9\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm10\n\t"
                       "vfmadd231pd 16(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm11\n\t"
                       "vfmadd231pd 16(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm12\n\t"
                       "vfmadd231pd 16(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm13\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm14\n\t"
                       "vfmadd231pd 16(%%r10)%{1to8%}, %%zmm0, %%zmm15\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 16(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 16(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 16(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 16(%%r10,%%r11,1)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,8)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 16(%%rdx)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 16(%%rdx,%%r15,1)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 16(%%rdx,%%r15,2)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 16(%%rdx,%%rax,1)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 16(%%rdx,%%r15,4)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 16(%%rdx,%%rbx,1)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 16(%%rdx,%%rax,2)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 16(%%rdx,%%r11,1)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovupd 4096(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 24(%%rsi)%{1to8%}, %%zmm1, %%zmm6\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm7\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm8\n\t"
                       "vfmadd231pd 24(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm9\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm10\n\t"
                       "vfmadd231pd 24(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm11\n\t"
                       "vfmadd231pd 24(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm12\n\t"
                       "vfmadd231pd 24(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm13\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm14\n\t"
                       "vfmadd231pd 24(%%r10)%{1to8%}, %%zmm1, %%zmm15\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 24(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 24(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 24(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 24(%%r10,%%r11,1)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,8)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 24(%%rdx)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 24(%%rdx,%%r15,1)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 24(%%rdx,%%r15,2)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 24(%%rdx,%%rax,1)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 24(%%rdx,%%r15,4)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 24(%%rdx,%%rbx,1)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 24(%%rdx,%%rax,2)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 24(%%rdx,%%r11,1)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "vmovupd 5120(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 32(%%rsi)%{1to8%}, %%zmm0, %%zmm6\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm7\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm8\n\t"
                       "vfmadd231pd 32(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm9\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm10\n\t"
                       "vfmadd231pd 32(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm11\n\t"
                       "vfmadd231pd 32(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm12\n\t"
                       "vfmadd231pd 32(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm13\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm14\n\t"
                       "vfmadd231pd 32(%%r10)%{1to8%}, %%zmm0, %%zmm15\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 32(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 32(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 32(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 32(%%r10,%%r11,1)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,8)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 32(%%rdx)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 32(%%rdx,%%r15,1)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 32(%%rdx,%%r15,2)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 32(%%rdx,%%rax,1)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 32(%%rdx,%%r15,4)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 32(%%rdx,%%rbx,1)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 32(%%rdx,%%rax,2)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 32(%%rdx,%%r11,1)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovupd 6144(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 40(%%rsi)%{1to8%}, %%zmm1, %%zmm6\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm7\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm8\n\t"
                       "vfmadd231pd 40(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm9\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm10\n\t"
                       "vfmadd231pd 40(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm11\n\t"
                       "vfmadd231pd 40(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm12\n\t"
                       "vfmadd231pd 40(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm13\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm14\n\t"
                       "vfmadd231pd 40(%%r10)%{1to8%}, %%zmm1, %%zmm15\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 40(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 40(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 40(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 40(%%r10,%%r11,1)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,8)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 40(%%rdx)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 40(%%rdx,%%r15,1)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 40(%%rdx,%%r15,2)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 40(%%rdx,%%rax,1)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 40(%%rdx,%%r15,4)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 40(%%rdx,%%rbx,1)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 40(%%rdx,%%rax,2)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 40(%%rdx,%%r11,1)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "vmovupd 7168(%%rdi), %%zmm1\n\t"
                       "addq $8192, %%rdi\n\t"
                       "vfmadd231pd 48(%%rsi)%{1to8%}, %%zmm0, %%zmm6\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm7\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm8\n\t"
                       "vfmadd231pd 48(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm9\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm10\n\t"
                       "vfmadd231pd 48(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm11\n\t"
                       "vfmadd231pd 48(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm12\n\t"
                       "vfmadd231pd 48(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm13\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm14\n\t"
                       "vfmadd231pd 48(%%r10)%{1to8%}, %%zmm0, %%zmm15\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 48(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 48(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 48(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 48(%%r10,%%r11,1)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,8)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 48(%%rdx)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 48(%%rdx,%%r15,1)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 48(%%rdx,%%r15,2)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 48(%%rdx,%%rax,1)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 48(%%rdx,%%r15,4)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 48(%%rdx,%%rbx,1)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 48(%%rdx,%%rax,2)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 48(%%rdx,%%r11,1)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "addq $8192, %%rdi\n\t"
                       "vfmadd231pd 56(%%rsi)%{1to8%}, %%zmm1, %%zmm6\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm7\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm8\n\t"
                       "vfmadd231pd 56(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm9\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm10\n\t"
                       "vfmadd231pd 56(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm11\n\t"
                       "vfmadd231pd 56(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm12\n\t"
                       "vfmadd231pd 56(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm13\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm14\n\t"
                       "vfmadd231pd 56(%%r10)%{1to8%}, %%zmm1, %%zmm15\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 56(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 56(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 56(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 56(%%r10,%%r11,1)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,8)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 56(%%rdx)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 56(%%rdx,%%r15,1)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 56(%%rdx,%%r15,2)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 56(%%rdx,%%rax,1)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 56(%%rdx,%%r15,4)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 56(%%rdx,%%rbx,1)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 56(%%rdx,%%rax,2)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 56(%%rdx,%%r11,1)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "subq $8192, %%rdi\n\t"
                       "addq $64, %%rsi\n\t"
                       "cmpq $128, %%r14\n\t"
                       "jl 2b\n\t"
                       "subq $1024, %%rsi\n\t"
                       "popq %%rdx\n\t"
                       "vmovupd %%zmm6, 0(%%rdx)\n\t"
                       "vmovupd %%zmm7, 2048(%%rdx)\n\t"
                       "vmovupd %%zmm8, 4096(%%rdx)\n\t"
                       "vmovupd %%zmm9, 6144(%%rdx)\n\t"
                       "vmovupd %%zmm10, 8192(%%rdx)\n\t"
                       "vmovupd %%zmm11, 10240(%%rdx)\n\t"
                       "vmovupd %%zmm12, 12288(%%rdx)\n\t"
                       "vmovupd %%zmm13, 14336(%%rdx)\n\t"
                       "vmovupd %%zmm14, 16384(%%rdx)\n\t"
                       "vmovupd %%zmm15, 18432(%%rdx)\n\t"
                       "vmovupd %%zmm16, 20480(%%rdx)\n\t"
                       "vmovupd %%zmm17, 22528(%%rdx)\n\t"
                       "vmovupd %%zmm18, 24576(%%rdx)\n\t"
                       "vmovupd %%zmm19, 26624(%%rdx)\n\t"
                       "vmovupd %%zmm20, 28672(%%rdx)\n\t"
                       "vmovupd %%zmm21, 30720(%%rdx)\n\t"
                       "vmovupd %%zmm22, 32768(%%rdx)\n\t"
                       "vmovupd %%zmm23, 34816(%%rdx)\n\t"
                       "vmovupd %%zmm24, 36864(%%rdx)\n\t"
                       "vmovupd %%zmm25, 38912(%%rdx)\n\t"
                       "vmovupd %%zmm26, 40960(%%rdx)\n\t"
                       "vmovupd %%zmm27, 43008(%%rdx)\n\t"
                       "vmovupd %%zmm28, 45056(%%rdx)\n\t"
                       "vmovupd %%zmm29, 47104(%%rdx)\n\t"
                       "vmovupd %%zmm30, 49152(%%rdx)\n\t"
                       "vmovupd %%zmm31, 51200(%%rdx)\n\t"
                       "addq $64, %%rdx\n\t"
                       "subq $131008, %%rdi\n\t"
                       "cmpq $128, %%r12\n\t"
                       "jl 1b\n\t"
                       "addq $52224, %%rdx\n\t"
                       "addq $26624, %%rsi\n\t"
                       "subq $1024, %%rdi\n\t"
                       "cmpq $78, %%r13\n\t"
                       "jl 0b\n\t"
                       "0:\n\t"
                       "addq $25, %%r13\n\t"
                       "movq $0, %%r12\n\t"
                       "1:\n\t"
                       "addq $8, %%r12\n\t"
                       "vmovupd 0(%%rdx), %%zmm7\n\t"
                       "vmovupd 2048(%%rdx), %%zmm8\n\t"
                       "vmovupd 4096(%%rdx), %%zmm9\n\t"
                       "vmovupd 6144(%%rdx), %%zmm10\n\t"
                       "vmovupd 8192(%%rdx), %%zmm11\n\t"
                       "vmovupd 10240(%%rdx), %%zmm12\n\t"
                       "vmovupd 12288(%%rdx), %%zmm13\n\t"
                       "vmovupd 14336(%%rdx), %%zmm14\n\t"
                       "vmovupd 16384(%%rdx), %%zmm15\n\t"
                       "vmovupd 18432(%%rdx), %%zmm16\n\t"
                       "vmovupd 20480(%%rdx), %%zmm17\n\t"
                       "vmovupd 22528(%%rdx), %%zmm18\n\t"
                       "vmovupd 24576(%%rdx), %%zmm19\n\t"
                       "vmovupd 26624(%%rdx), %%zmm20\n\t"
                       "vmovupd 28672(%%rdx), %%zmm21\n\t"
                       "vmovupd 30720(%%rdx), %%zmm22\n\t"
                       "vmovupd 32768(%%rdx), %%zmm23\n\t"
                       "vmovupd 34816(%%rdx), %%zmm24\n\t"
                       "vmovupd 36864(%%rdx), %%zmm25\n\t"
                       "vmovupd 38912(%%rdx), %%zmm26\n\t"
                       "vmovupd 40960(%%rdx), %%zmm27\n\t"
                       "vmovupd 43008(%%rdx), %%zmm28\n\t"
                       "vmovupd 45056(%%rdx), %%zmm29\n\t"
                       "vmovupd 47104(%%rdx), %%zmm30\n\t"
                       "vmovupd 49152(%%rdx), %%zmm31\n\t"
                       "pushq %%rdx\n\t"
                       "movq $0, %%r14\n\t"
                       "2:\n\t"
                       "addq $8, %%r14\n\t"
                       "movq $1024, %%r15\n\t"
                       "movq $3072, %%rax\n\t"
                       "movq $5120, %%rbx\n\t"
                       "movq $7168, %%r11\n\t"
                       "movq %%rsi, %%r10\n\t"
                       "addq $9216, %%r10\n\t"
                       "movq %%rsi, %%rdx\n\t"
                       "addq $18432, %%rdx\n\t"
                       "vmovupd 0(%%rdi), %%zmm0\n\t"
                       "vmovupd 1024(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 0(%%rsi)%{1to8%}, %%zmm0, %%zmm7\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm8\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm9\n\t"
                       "vfmadd231pd 0(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm10\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm11\n\t"
                       "vfmadd231pd 0(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm12\n\t"
                       "vfmadd231pd 0(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm13\n\t"
                       "vfmadd231pd 0(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm14\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm15\n\t"
                       "vfmadd231pd 0(%%r10)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 0(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 0(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 0(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 0(%%r10,%%r11,1)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 0(%%r10,%%r15,8)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 0(%%rdx)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 0(%%rdx,%%r15,1)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 0(%%rdx,%%r15,2)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 0(%%rdx,%%rax,1)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 0(%%rdx,%%r15,4)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 0(%%rdx,%%rbx,1)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 0(%%rdx,%%rax,2)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovupd 2048(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 8(%%rsi)%{1to8%}, %%zmm1, %%zmm7\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm8\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm9\n\t"
                       "vfmadd231pd 8(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm10\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm11\n\t"
                       "vfmadd231pd 8(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm12\n\t"
                       "vfmadd231pd 8(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm13\n\t"
                       "vfmadd231pd 8(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm14\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm15\n\t"
                       "vfmadd231pd 8(%%r10)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 8(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 8(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 8(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 8(%%r10,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 8(%%r10,%%r15,8)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 8(%%rdx)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 8(%%rdx,%%r15,1)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 8(%%rdx,%%r15,2)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 8(%%rdx,%%rax,1)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 8(%%rdx,%%r15,4)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 8(%%rdx,%%rbx,1)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 8(%%rdx,%%rax,2)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "vmovupd 3072(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 16(%%rsi)%{1to8%}, %%zmm0, %%zmm7\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm8\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm9\n\t"
                       "vfmadd231pd 16(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm10\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm11\n\t"
                       "vfmadd231pd 16(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm12\n\t"
                       "vfmadd231pd 16(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm13\n\t"
                       "vfmadd231pd 16(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm14\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm15\n\t"
                       "vfmadd231pd 16(%%r10)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 16(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 16(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 16(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 16(%%r10,%%r11,1)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 16(%%r10,%%r15,8)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 16(%%rdx)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 16(%%rdx,%%r15,1)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 16(%%rdx,%%r15,2)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 16(%%rdx,%%rax,1)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 16(%%rdx,%%r15,4)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 16(%%rdx,%%rbx,1)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 16(%%rdx,%%rax,2)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovupd 4096(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 24(%%rsi)%{1to8%}, %%zmm1, %%zmm7\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm8\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm9\n\t"
                       "vfmadd231pd 24(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm10\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm11\n\t"
                       "vfmadd231pd 24(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm12\n\t"
                       "vfmadd231pd 24(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm13\n\t"
                       "vfmadd231pd 24(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm14\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm15\n\t"
                       "vfmadd231pd 24(%%r10)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 24(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 24(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 24(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 24(%%r10,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 24(%%r10,%%r15,8)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 24(%%rdx)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 24(%%rdx,%%r15,1)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 24(%%rdx,%%r15,2)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 24(%%rdx,%%rax,1)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 24(%%rdx,%%r15,4)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 24(%%rdx,%%rbx,1)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 24(%%rdx,%%rax,2)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "vmovupd 5120(%%rdi), %%zmm1\n\t"
                       "vfmadd231pd 32(%%rsi)%{1to8%}, %%zmm0, %%zmm7\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm8\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm9\n\t"
                       "vfmadd231pd 32(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm10\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm11\n\t"
                       "vfmadd231pd 32(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm12\n\t"
                       "vfmadd231pd 32(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm13\n\t"
                       "vfmadd231pd 32(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm14\n\t"
                       "vfmadd231pd 32(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm15\n\t"
                       "vfmadd231pd 32(%%r10)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 32(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 32(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 32(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 32(%%r10,%%r11,1)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 32(%%r10,%%r15,8)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 32(%%rdx)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 32(%%rdx,%%r15,1)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 32(%%rdx,%%r15,2)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 32(%%rdx,%%rax,1)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 32(%%rdx,%%r15,4)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 32(%%rdx,%%rbx,1)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 32(%%rdx,%%rax,2)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovupd 6144(%%rdi), %%zmm0\n\t"
                       "vfmadd231pd 40(%%rsi)%{1to8%}, %%zmm1, %%zmm7\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm8\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm9\n\t"
                       "vfmadd231pd 40(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm10\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm11\n\t"
                       "vfmadd231pd 40(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm12\n\t"
                       "vfmadd231pd 40(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm13\n\t"
                       "vfmadd231pd 40(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm14\n\t"
                       "vfmadd231pd 40(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm15\n\t"
                       "vfmadd231pd 40(%%r10)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 40(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 40(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 40(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 40(%%r10,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 40(%%r10,%%r15,8)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 40(%%rdx)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 40(%%rdx,%%r15,1)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 40(%%rdx,%%r15,2)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 40(%%rdx,%%rax,1)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 40(%%rdx,%%r15,4)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 40(%%rdx,%%rbx,1)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 40(%%rdx,%%rax,2)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "vmovupd 7168(%%rdi), %%zmm1\n\t"
                       "addq $8192, %%rdi\n\t"
                       "vfmadd231pd 48(%%rsi)%{1to8%}, %%zmm0, %%zmm7\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm8\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm9\n\t"
                       "vfmadd231pd 48(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm10\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,4)%{1to8%}, %%zmm0, %%zmm11\n\t"
                       "vfmadd231pd 48(%%rsi,%%rbx,1)%{1to8%}, %%zmm0, %%zmm12\n\t"
                       "vfmadd231pd 48(%%rsi,%%rax,2)%{1to8%}, %%zmm0, %%zmm13\n\t"
                       "vfmadd231pd 48(%%rsi,%%r11,1)%{1to8%}, %%zmm0, %%zmm14\n\t"
                       "vfmadd231pd 48(%%rsi,%%r15,8)%{1to8%}, %%zmm0, %%zmm15\n\t"
                       "vfmadd231pd 48(%%r10)%{1to8%}, %%zmm0, %%zmm16\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,1)%{1to8%}, %%zmm0, %%zmm17\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,2)%{1to8%}, %%zmm0, %%zmm18\n\t"
                       "vfmadd231pd 48(%%r10,%%rax,1)%{1to8%}, %%zmm0, %%zmm19\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,4)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 48(%%r10,%%rbx,1)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 48(%%r10,%%rax,2)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 48(%%r10,%%r11,1)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 48(%%r10,%%r15,8)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 48(%%rdx)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vfmadd231pd 48(%%rdx,%%r15,1)%{1to8%}, %%zmm0, %%zmm26\n\t"
                       "vfmadd231pd 48(%%rdx,%%r15,2)%{1to8%}, %%zmm0, %%zmm27\n\t"
                       "vfmadd231pd 48(%%rdx,%%rax,1)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 48(%%rdx,%%r15,4)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 48(%%rdx,%%rbx,1)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 48(%%rdx,%%rax,2)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "addq $8192, %%rdi\n\t"
                       "vfmadd231pd 56(%%rsi)%{1to8%}, %%zmm1, %%zmm7\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm8\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm9\n\t"
                       "vfmadd231pd 56(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm10\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,4)%{1to8%}, %%zmm1, %%zmm11\n\t"
                       "vfmadd231pd 56(%%rsi,%%rbx,1)%{1to8%}, %%zmm1, %%zmm12\n\t"
                       "vfmadd231pd 56(%%rsi,%%rax,2)%{1to8%}, %%zmm1, %%zmm13\n\t"
                       "vfmadd231pd 56(%%rsi,%%r11,1)%{1to8%}, %%zmm1, %%zmm14\n\t"
                       "vfmadd231pd 56(%%rsi,%%r15,8)%{1to8%}, %%zmm1, %%zmm15\n\t"
                       "vfmadd231pd 56(%%r10)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 56(%%r10,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,4)%{1to8%}, %%zmm1, %%zmm20\n\t"
                       "vfmadd231pd 56(%%r10,%%rbx,1)%{1to8%}, %%zmm1, %%zmm21\n\t"
                       "vfmadd231pd 56(%%r10,%%rax,2)%{1to8%}, %%zmm1, %%zmm22\n\t"
                       "vfmadd231pd 56(%%r10,%%r11,1)%{1to8%}, %%zmm1, %%zmm23\n\t"
                       "vfmadd231pd 56(%%r10,%%r15,8)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 56(%%rdx)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 56(%%rdx,%%r15,1)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 56(%%rdx,%%r15,2)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 56(%%rdx,%%rax,1)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 56(%%rdx,%%r15,4)%{1to8%}, %%zmm1, %%zmm29\n\t"
                       "vfmadd231pd 56(%%rdx,%%rbx,1)%{1to8%}, %%zmm1, %%zmm30\n\t"
                       "vfmadd231pd 56(%%rdx,%%rax,2)%{1to8%}, %%zmm1, %%zmm31\n\t"
                       "subq $8192, %%rdi\n\t"
                       "addq $64, %%rsi\n\t"
                       "cmpq $128, %%r14\n\t"
                       "jl 2b\n\t"
                       "subq $1024, %%rsi\n\t"
                       "popq %%rdx\n\t"
                       "vmovupd %%zmm7, 0(%%rdx)\n\t"
                       "vmovupd %%zmm8, 2048(%%rdx)\n\t"
                       "vmovupd %%zmm9, 4096(%%rdx)\n\t"
                       "vmovupd %%zmm10, 6144(%%rdx)\n\t"
                       "vmovupd %%zmm11, 8192(%%rdx)\n\t"
                       "vmovupd %%zmm12, 10240(%%rdx)\n\t"
                       "vmovupd %%zmm13, 12288(%%rdx)\n\t"
                       "vmovupd %%zmm14, 14336(%%rdx)\n\t"
                       "vmovupd %%zmm15, 16384(%%rdx)\n\t"
                       "vmovupd %%zmm16, 18432(%%rdx)\n\t"
                       "vmovupd %%zmm17, 20480(%%rdx)\n\t"
                       "vmovupd %%zmm18, 22528(%%rdx)\n\t"
                       "vmovupd %%zmm19, 24576(%%rdx)\n\t"
                       "vmovupd %%zmm20, 26624(%%rdx)\n\t"
                       "vmovupd %%zmm21, 28672(%%rdx)\n\t"
                       "vmovupd %%zmm22, 30720(%%rdx)\n\t"
                       "vmovupd %%zmm23, 32768(%%rdx)\n\t"
                       "vmovupd %%zmm24, 34816(%%rdx)\n\t"
                       "vmovupd %%zmm25, 36864(%%rdx)\n\t"
                       "vmovupd %%zmm26, 38912(%%rdx)\n\t"
                       "vmovupd %%zmm27, 40960(%%rdx)\n\t"
                       "vmovupd %%zmm28, 43008(%%rdx)\n\t"
                       "vmovupd %%zmm29, 45056(%%rdx)\n\t"
                       "vmovupd %%zmm30, 47104(%%rdx)\n\t"
                       "vmovupd %%zmm31, 49152(%%rdx)\n\t"
                       "addq $64, %%rdx\n\t"
                       "subq $131008, %%rdi\n\t"
                       "cmpq $128, %%r12\n\t"
                       "jl 1b\n\t"
                       "addq $50176, %%rdx\n\t"
                       "addq $25600, %%rsi\n\t"
                       "subq $1024, %%rdi\n\t"
                       "cmpq $128, %%r13\n\t"
                       "jl 0b\n\t"
                       : : "m"(A), "m"(B), "m"(C) : "k1","rax","rbx","rcx","rdx","rdi","rsi","r8","r9","r10","r11","r12","r13","r14","r15","zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15","zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");
#else
#pragma message ("LIBXSMM KERNEL COMPILATION ERROR in: " __FILE__)
#error No kernel was compiled, lacking support for current architecture?
#endif
}

// S 8, else 4
void microkernel_gen_8(const double* A, const double* B, double* C) {
#ifdef __AVX512F__
  __asm__ __volatile__("movq %0, %%rdi\n\t"
                       "movq %1, %%rsi\n\t"
                       "movq %2, %%rdx\n\t"
                       "movq $0, %%r12\n\t"
                       "movq $0, %%r13\n\t"
                       "movq $0, %%r14\n\t"
                       "movq $15, %%r9\n\t"
                       "kmovw %%r9d, %%k1\n\t"
                       "vmovupd 0(%%rdx), %%zmm28%{%%k1%}%{z%}\n\t"
                       "vmovupd 64(%%rdx), %%zmm29%{%%k1%}%{z%}\n\t"
                       "vmovupd 128(%%rdx), %%zmm30%{%%k1%}%{z%}\n\t"
                       "vmovupd 192(%%rdx), %%zmm31%{%%k1%}%{z%}\n\t"
                       "movq $32, %%r15\n\t"
                       "movq $96, %%rax\n\t"
                       "movq $160, %%rbx\n\t"
                       "movq $224, %%r11\n\t"
                       "vpxord %%zmm24, %%zmm24, %%zmm24\n\t"
                       "vpxord %%zmm25, %%zmm25, %%zmm25\n\t"
                       "vpxord %%zmm26, %%zmm26, %%zmm26\n\t"
                       "vpxord %%zmm27, %%zmm27, %%zmm27\n\t"
                       "vpxord %%zmm20, %%zmm20, %%zmm20\n\t"
                       "vpxord %%zmm21, %%zmm21, %%zmm21\n\t"
                       "vpxord %%zmm22, %%zmm22, %%zmm22\n\t"
                       "vpxord %%zmm23, %%zmm23, %%zmm23\n\t"
                       "vpxord %%zmm16, %%zmm16, %%zmm16\n\t"
                       "vpxord %%zmm17, %%zmm17, %%zmm17\n\t"
                       "vpxord %%zmm18, %%zmm18, %%zmm18\n\t"
                       "vpxord %%zmm19, %%zmm19, %%zmm19\n\t"
                       "vmovupd 0(%%rdi), %%zmm0%{%%k1%}%{z%}\n\t"
                       "vmovupd 32(%%rdi), %%zmm1%{%%k1%}%{z%}\n\t"
                       "vfmadd231pd 0(%%rsi)%{1to8%}, %%zmm0, %%zmm28\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 0(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovupd 64(%%rdi), %%zmm0%{%%k1%}%{z%}\n\t"
                       "vfmadd231pd 8(%%rsi)%{1to8%}, %%zmm1, %%zmm24\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm25\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 8(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vmovupd 96(%%rdi), %%zmm1%{%%k1%}%{z%}\n\t"
                       "vfmadd231pd 16(%%rsi)%{1to8%}, %%zmm0, %%zmm20\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm21\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm22\n\t"
                       "vfmadd231pd 16(%%rsi,%%rax,1)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 24(%%rsi)%{1to8%}, %%zmm1, %%zmm16\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm17\n\t"
                       "vfmadd231pd 24(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm18\n\t"
                       "vfmadd231pd 24(%%rsi,%%rax,1)%{1to8%}, %%zmm1, %%zmm19\n\t"
                       "vaddpd %%zmm24, %%zmm28, %%zmm28\n\t"
                       "vaddpd %%zmm25, %%zmm29, %%zmm29\n\t"
                       "vaddpd %%zmm26, %%zmm30, %%zmm30\n\t"
                       "vaddpd %%zmm27, %%zmm31, %%zmm31\n\t"
                       "vaddpd %%zmm20, %%zmm28, %%zmm28\n\t"
                       "vaddpd %%zmm21, %%zmm29, %%zmm29\n\t"
                       "vaddpd %%zmm22, %%zmm30, %%zmm30\n\t"
                       "vaddpd %%zmm23, %%zmm31, %%zmm31\n\t"
                       "vaddpd %%zmm16, %%zmm28, %%zmm28\n\t"
                       "vaddpd %%zmm17, %%zmm29, %%zmm29\n\t"
                       "vaddpd %%zmm18, %%zmm30, %%zmm30\n\t"
                       "vaddpd %%zmm19, %%zmm31, %%zmm31\n\t"
                       "vmovupd %%zmm28, 0(%%rdx)%{%%k1%}\n\t"
                       "vmovupd %%zmm29, 64(%%rdx)%{%%k1%}\n\t"
                       "vmovupd %%zmm30, 128(%%rdx)%{%%k1%}\n\t"
                       "vmovupd %%zmm31, 192(%%rdx)%{%%k1%}\n\t"
                       "addq $32, %%rdx\n\t"
                       "addq $32, %%rdi\n\t"
                       : : "m"(A), "m"(B), "m"(C) : "k1","rax","rbx","rcx","rdx","rdi","rsi","r8","r9","r10","r11","r12","r13","r14","r15","zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15","zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");
#else
#pragma message ("LIBXSMM KERNEL COMPILATION ERROR in: " __FILE__)
#error No kernel was compiled, lacking support for current architecture?
#endif

}



////////////

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

/**
 * A: MC x K, ld S
 * B:  K x S, ld K
 * C: MC x S, ld S
 */
void GEBP(double* A, double* B, double* C, double* A_pack, int S, int threadsPerTeam)
{
	
	// pack A
	pack_A(A, A_pack, S);
	#pragma omp parallel for num_threads(threadsPerTeam)
	for (int n = 0; n < S; n += N) {
		
		
		//int tid = omp_get_thread_num();
        //printf("Level %d: number of threads in the team %d- %d\n",
        //          2,  tid,omp_get_num_threads());
		
		
		//microkernel
		//microkernel_gen(A, &B[n*S], &C[n*S]);
		//microkernel_gen(A_pack, &B[n*K], &C[n*S]);  // works for square
		//microkernel_gen_256(A_pack, &B[n*K], &C[n*S]);  // wrong result
		//microkernel_gen_8(A_pack, &B[n*K], &C[n*S]);  // wrong result
		microkernel_gen2(A_pack, &B[n*K], &C[n*S], S); 
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
        //          1,  tid,omp_get_num_threads());
		
		
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
    
    
 //dgemm_opt(A,B,C);
 //microkernel_gen(A,B,C);
}


int main(int argc, char** argv) {
  int S = 4096;
  bool test = true;
  int threadsPerTeam = 1;
  int nRepeat = 10;
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
      A[j*S + i] = i + j;
      B[j*S + i] = (S-i) + (S-j);
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

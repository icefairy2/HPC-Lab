#ifndef KERNELS_H
#define KERNELS_H

void kernel_M8_N8_K256_S512_knl(const double* A, const double* B, double* C);
void kernel_M8_N8_K256_S512_noarch(const double* A, const double* B, double* C);
void kernel_M2_N2_K4_S8_noarch(const double* A, const double* B, double* C);
void kernel_M4_N2_K4_S8_noarch(const double* A, const double* B, double* C);
void kernel_M64_N8_K256_S1024_knl(const double* A, const double* B, double* C);
void kernel_M128_N8_K256_S1024_knl(const double* A, const double* B, double* C);
void kernel_M128_N128_K256_S1024_knl(const double* A, const double* B, double* C);

void kernel_M128_N128_K128_S256_knl(const double* A, const double* B, double* C);
void kernel_M128_N128_K128_S512_knl(const double* A, const double* B, double* C);
void kernel_M128_N128_K128_S1024_knl(const double* A, const double* B, double* C);
void kernel_M128_N128_K128_S2048_knl(const double* A, const double* B, double* C);
void kernel_M128_N128_K128_S4096_knl(const double* A, const double* B, double* C);
void kernel_M128_N128_K128_S8192_knl(const double* A, const double* B, double* C);

void kernel_M8_N8_K512_S1024_knl(const double* A, const double* B, double* C);

#endif
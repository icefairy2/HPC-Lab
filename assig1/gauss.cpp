/* row wise gaus algorithm
 * pattern for practical course
 * -------------------------
 * autor: Markus Brenk
 * date: 2002-09-25
 * =================================================== */

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <limits>
#include "Stopwatch.h"

/** print a 3x3 matrix */
void print_matrix(const char* name, double matrix[3][3]);
/** print a 3d vector */
void print_vector(const char* name, double vec[3]);
/**
 *  initialisation: generates the following LGS:
 *  ( 3  1  1)       (5r)                         (r)
 *  ( 1  4  1) * X = (6r)        => solution  X = (r)
 *  ( 1  1  5)       (7r)                         (r)
 */
void init(double r, double A[3][3], double b[3], double x[3]);
/** check if x_d == r */
bool check(double r, double x[3]);
/** performs gauss elimination */
void gauss_elimination(double A[3][3], double b[3], double x[3]);

void init(double A[3][3]);
void init(double L[3][3], double U[3][3]);
void init(double r, double b[3], double x[3]) ;
void solve(double L[3][3], double U[3][3], double b[3], double x[3]);

int main() {
  double A[NRHS][3][3];
  double b[NRHS][3];
  double x[NRHS][3];
  
  double L[3][3]{0};
  double U[3][3]{0};
  
  init(L, U);

  for (int r = 0; r < NRHS; ++r) {
#if VERS == 1
    init(static_cast<double>(r)/NRHS, A[r], b[r], x[r]);
#else
    init(r, b[r], x[r]);
#endif
  }

  Stopwatch stopwatch;
  stopwatch.start();

  for (int r = 0; r < NRHS; ++r) {
#if VERS == 1
    gauss_elimination(A[r], b[r], x[r]);
#else
    solve(L, U, b[r], x[r]);
#endif
  }

  double time = stopwatch.stop();
  printf("Time: %lf us\n", time * 1.0e6);
  
  bool correct = true;
  for (int r = 2; r < NRHS; r++) {
    correct = correct && check(static_cast<double>(r)/NRHS, x[r]);
  }
  if (!correct) {
    printf("Incorrect code.\n");
  }
  
  return 0;
}

void init(double A[3][3])
{
    int n = 3;

    for (int j = 0; j < n; j++) {
        A[j][j]=(float)(n-1)+(float)(j+1);

        for (int i = j+1; i < n; i++)
            A[i][j]=1.;
        for (int i = 0; i < j; i++)
            A[i][j]=1.;
    }
}

void init(double L[3][3], double U[3][3])
{
    int n = 3;
    init(U);
    for (int i = 0; i < n; i++)
        L[i][i] = 1;
    
    for (int i = 0; i < n - 1; i++)
        for (int j = i+1; j < n; j++) {
            L[j][i] = U[j][i] / U[i][i];
            
            for (int k = i; k < n; k++)
                U[j][k] -= L[j][i] * U[i][k];
        }
}

void init(double r, double b[3], double x[3]) 
{
    int n = 3;
    for (int j = 0;j < n; j++) {
        b[j]=r * ((float)(2*n-2)+(float)(j+1));
        x[j]=0.;
    }
}

void init(double r, double A[3][3], double b[3], double x[3]) {
    int i,j;
    int n = 3;

    for (j=0;j<n;j++)
    {
        b[j]=r * ((float)(2*n-2)+(float)(j+1));
        A[j][j]=(float)(n-1)+(float)(j+1);
        x[j]=0.;

        for (i=j+1;i<n;i++) {
            A[i][j]=1.;
        }
        for (i=0;i<j;i++) {
            A[i][j]=1.;
        }
    }
}

bool check(double r, double x[3]) {
  for (int d = 0; d < 3; ++d) {
    if (fabs(x[d] - r*1.0) > 10.0*std::numeric_limits<double>::epsilon()) {
        printf("diff %lf %lf\n", x[d], r);
      return false;
    }
  }
  return true;
}

inline void solve(double L[3][3], double U[3][3], double b[3], double x[3])
{
    int n = 3;
    double y[3]{0};

    for (int i = 0; i < n; i++) {
        y[i] = b[i];

        #pragma vector always
        for (int j = 0; j < i; j++)
            y[i] -= L[i][j]*y[j];
        y[i] /= L[i][i];
    }

    for (int i = n-1; i >= 0; i--) {
        x[i] = y[i];

        #pragma vector always
        for (int j = i+1; j < n; j++)
            x[i] -= U[i][j]*x[j];
        x[i] /= U[i][i];
    }
}

void gauss_elimination(double A[3][3], double b[3], double x[3]) {
    int n = 3;
    int i,j,k;

    for (i = 0; i < n; i++) {

        for (j = i+1; j < n; j++) {
            A[i][j] = A[i][j] / A[i][i];
        }
        b[i] = b[i] / A[i][i];

        for (j = i+1; j < n; j++) {
            double factor = A[j][i];
            for (k = i; k < n; k++) {
                A[j][k] = A[j][k] - A[i][k] * factor;
            }
            b[j] = b[j] - factor * b[i];
        }
    }


    for (i = n-1; i >= 0; i--) {
        x[i] = b[i];
        for(j = i+1; j < n; j++) {
            x[i] -= A[i][j] * x[j];
        }
    }
}

void print_matrix(const char* name, double matrix[3][3]) {
    int i, j;
    printf("Matrix %s: \n", name);
    for (i = 0; i < 3; i++) {
        for (j = 0; j < 3; j++) {
            printf(" %f ", matrix[i][j]);
        }
        printf(" ;\n");
    }
}

void print_vector(const char* name, double vec[3]) {
    int i;
    printf("vector %s: \n", name);
    for (i=0;i<3;i++)
    {
        printf(" %f \n",vec[i]);
    }
}


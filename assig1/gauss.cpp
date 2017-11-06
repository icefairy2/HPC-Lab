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
/** check if x_d == r */
bool check(double r, double x[3]);

int main() {
  double A[3][3][NRHS];
  double b[3][NRHS];
  double x[3][NRHS];
  double checkX[NRHS][3];
  int n = 3;

  for (int r = 0; r < NRHS; ++r) {
    for (int j=0;j<n;j++)
    {
        b[j][r]=static_cast<double>(r)/NRHS * ((float)(2*n-2)+(float)(j+1));
        A[j][j][r]=(float)(n-1)+(float)(j+1);
        x[j][r]=0.;

        for (int i=j+1;i<n;i++) {
            A[i][j][r]=1.;
        }
        for (int i=0;i<j;i++) {
            A[i][j][r]=1.;
        }
    }
  }

  Stopwatch stopwatch;
  stopwatch.start();


  for (int i = 0; i < n; i++) {

    for (int j = i+1; j < n; j++) {
        for (int r = 0; r < NRHS; ++r)
            A[i][j][r] = A[i][j][r] / A[i][i][r];
    }
    for (int r = 0; r < NRHS; ++r)
        b[i][r] = b[i][r] / A[i][i][r];

    for (int j = i+1; j < n; j++) {
        double factor[NRHS];
        for (int r = 0; r < NRHS; ++r)
            factor[r] = A[j][i][r];

        for (int k = i; k < n; k++) {
            for (int r = 0; r < NRHS; ++r)
                A[j][k][r] = A[j][k][r] - A[i][k][r] * factor[r];
        }
        for (int r = 0; r < NRHS; ++r)
            b[j][r] = b[j][r] - factor[r] * b[i][r];
    }
  }


    for (int i = n-1; i >= 0; i--) {
        for (int r = 0; r < NRHS; ++r)
            x[i][r] = b[i][r];
        for(int j = i+1; j < n; j++) {
            for (int r = 0; r < NRHS; ++r)
                x[i][r] -= A[i][j][r] * x[j][r];
        }
    }

  double time = stopwatch.stop();
  printf("Time: %lf us\n", time * 1.0e6);

  for (int i = 0; i < NRHS; i++)
    for (int j = 0; j < n; j++)
        checkX[i][j] = x[j][i];


  
  bool correct = true;
  for (int r = 0; r < NRHS; r++) {
    correct = correct && check(static_cast<double>(r)/NRHS, checkX[r]);
  }
  if (!correct) {
    printf("Incorrect code.\n");
  }
  
  return 0;
}

bool check(double r, double x[3]) {
  for (int d = 0; d < 3; ++d) {
    if (fabs(x[d] - r*1.0) > 10.0*std::numeric_limits<double>::epsilon()) {
      return false;
    }
  }
  return true;
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


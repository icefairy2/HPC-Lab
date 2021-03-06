#include <omp.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

//function to be integrated
double phi(double x) {
    return 1 / (1 + x*x);
}

int main(int argc, char** argv)
{
    //record starting time
    auto start_time = std::chrono::high_resolution_clock::now();

    int i;
    double h, y, sum;
    //number of partitions
    long n = 100000000;

    std::cout << "Number of partitions: " << n << std::endl;

    h = 1. / n;

    sum = 0;
#pragma omp parallel for private(y), shared(sum) 
    for (i = 0; i <= n; i++)
    {
        //calculate function value at current partition
        y = 1 / (1 + (i*h)*(i*h));
#pragma omp critical 
        //add current function value to sum
        sum += y;
    }

    sum *= 4. * h; //value of pi
    std::cout << "Result of integration is: " << sum << std::endl;

    auto current_time = std::chrono::high_resolution_clock::now();
    long diff;
    diff = std::chrono::duration_cast<std::chrono::milliseconds> (current_time - start_time).count();
    std::cout << "Execution time: " << diff << std::endl;
    return 0;
}
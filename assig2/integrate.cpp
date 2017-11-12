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

    h = 1. / n;

    sum = 0;

    for (i = 0; i <= n; i++)
    {
        //calculate function value at current partition
        y = phi(i*h);
        //add current function value to sum
        sum += y;
    }

    sum *= 4. * h;
    std::cout << "Result of integration is: " << sum << std::endl;

    auto current_time = std::chrono::high_resolution_clock::now();
    long diff;
    diff = std::chrono::duration_cast<std::chrono::microseconds> (current_time - start_time).count();
    std::cout << diff << std::endl;
    return 0;
}
/**
 * Quicksort implementation for practical course
 **/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Stopwatch.h"

#include <iostream>

void print_list(double *data, int length){
	int i;
	for(i = 0; i < length; i++)
		printf("%e\t", data[i]);
	printf("\n");
}

#define THRESHOLD 16

void quicksort(double *data, int length, int level){
	if (length <= 1) return;

	//print_list(data, length);

	double pivot = data[0];
	double temp;
	int left = 1;
	int right = length - 1;

	do {
		while(left < (length - 1) && data[left] <= pivot) left++;
		while(right > 0 && data[right] >= pivot) right--;

		/* swap elements */
		if(left < right){
			temp = data[left];
			data[left] = data[right];
			data[right] = temp;
		}

	} while(left < right);

	if(data[right] < pivot){
		data[0] = data[right];
		data[right] = pivot;
	}

	//print_list(data, length);

	/* recursion */
	#pragma omp task final(level >= THRESHOLD)
	quicksort(data, right, level + 1);
	#pragma omp task final(level >= THRESHOLD)
	quicksort(&(data[left]), length - left, level + 1);
}

int check(double *data, int length){
	int i;
	for(i = 1; i < length; i++)
		if(data[i] < data[i-1]) return 1;
	return 0;
}

int main(int argc, char **argv)
{
	int length;
	double *data;

	int mem_size;

	int i, j, k;

	length = 10000000;
	if(argc > 1){
		length = atoi(argv[1]);
	}

	data = (double*)malloc(length * sizeof(double));
	if(0 == data){
		printf("memory allocation failed");
		return 0;
	}

	/* initialisation */
	srand(0);
	for (i = 0; i < length; i++){
		data[i] = (double)rand() / (double)RAND_MAX;
	}

	Stopwatch stopwatch;
	stopwatch.start();

	//print_list(data, length);

  #pragma omp parallel
	#pragma omp single
	quicksort(data, length, 0);

	double time = stopwatch.stop();

	//print_list(data, length);

	printf("Size of dataset: %d, elapsed time[s] %e \n", length, time);

	if (check(data, length) != 0) {
		printf("Quicksort incorrect.\n");
	}

	return(0);
}

#ifndef CMATRIX_H
#define CMATRIX_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>

#include <errorHandling.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>

template <typename T>
bool vectorContains(std::vector<T> &vector, T value);

void printMatrix(double *array, int rows, int columns, int rank);
void generateMatrix(double *array, const int rows, const int columns);
void printInLine(double *array, int rows, int columns, int rank);
double* copyMatrix(double *matrix1, int rows, int columns);
void writeMatrixToFile(double* matrix, const int rows, const int columns, const std::string& filename);

void writeMatrixToFileColumnMajor(double* matrix, const int rows, const int columns, const std::string& filename);
void printMatrixColumnMajor(double *array, const int rows, const int columns, int rank);
void generateMatrixColumnMajor(double *array, const int rows, const int columns);

/* GPU Related functions */
__device__ double device_dabs(double value);
__device__ long int get_tid();
__global__ void initializeRandom(curandState *state, unsigned long seed);
__global__ void generateMatrix(double* array, int rows, int columns, curandState *state);
void generateMatrixGPU(double* array, const int rows, const int columns);

#endif
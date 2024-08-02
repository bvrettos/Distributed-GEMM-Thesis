#ifndef CMATRIX_H
#define CMATRIX_H

#include <iostream>
#include <string>
#include <fstream>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include "errorHandling.hpp"
#include "generalUtilities.hpp"

#include <cstring>
#include <type_traits>
#include <memory>
#include <utility>
#include <algorithm>

/* Print related functions */
void printMatrix(double *array, const long long rows,  const long long columns, int rank);
void printMatrixColumnMajor(double *array,  const long long rows,  const long long columns, int rank);
void printInLine(double *array,  const long long rows,  const long long columns, int rank);

/* Write matrix to file functions*/
void writeMatrixToFile(double* matrix, const int rows, const int columns, const std::string& filename);
void writeMatrixToFileColumnMajor(double* matrix,  const long long rows,  const long long columns, const std::string& filename);

/* Copy and generate matrices functions */
template <typename T>
T* copyMatrix(T *matrix,  const long long rows,  const long long columns);

template <typename T>
void generateMatrix(T *array,  const long long rows,  const long long columns);
template <typename T>
void generateMatrix(T *array,  const long long size);

template <typename T>
void generateMatrixGPU(T* array,  const long long rows,  const long long columns, const int deviceID);
template <typename T>
void generateMatrixGPU(T* array,  const long long size, const int deviceID);

template <typename T>
void MatrixInit(T *matrix , const long long rows, const long long columns, int loc);
template <typename T>
void MatrixInit(T *matrix , const long long size, int loc);

#endif
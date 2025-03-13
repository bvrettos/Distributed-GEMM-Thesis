#ifndef MATRIX_UTILITIES_HPP
#define MATRIX_UTILITIES_HPP

#include <cstdio>
#include <cassert>
#include <cstring>

#include <iostream>
#include <string>
#include <fstream>
#include <type_traits>
#include <memory>
#include <utility>
#include <algorithm>

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <time.h>
#include "errorHandling.hpp"
#include "generalUtilities.hpp"

#include <Enums.hpp>

/* Matrix I/O */
template <typename scalar_t>
void printMatrix(scalar_t* array, long long rows, long long columns, int rank = 0, MatrixLayout layout = MatrixLayout::ColumnMajor);

template <typename scalar_t>
void writeMatrix(scalar_t* array, long long rows, long long columns, std::string& filename, MatrixLayout layout = MatrixLayout::ColumnMajor);

/* Matrix Generation (CPU, GPU) */
template <typename scalar_t>
void generateMatrix(scalar_t *array, const long long rows, const long long columns);

template <typename scalar_t>
void generateMatrix(scalar_t *array, const long long size);

template <typename scalar_t>
void generateMatrixGPU(scalar_t *array, const long long rows, const long long columns, const int deviceID);

template <typename scalar_t>
void generateMatrixGPU(scalar_t *array, const long long size, const int deviceID);

#endif
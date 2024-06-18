#ifndef GENERAL_UTILITIES_HPP
#define GENERAL_UTILITIES_HPP

#include <cstdio>
#include <cstdlib>
#include <cblas.h>
#include <cublas_v2.h>
#include <typeinfo>
#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <errorHandling.hpp>

int getSlurmNumNodes();
double calculateGflops(const long long M, const long long N, const long long K, const double executionTime);

char cblasTransOpToChar(CBLAS_TRANSPOSE operation);
CBLAS_TRANSPOSE charToCblasTransOp(char operation);

char cublasTransOpToChar(cublasOperation_t operation);
cublasOperation_t charToCublasTransOp(char operation);

void getGPUMemoryInfo(long long *freeMemory, long long *maxMemory, const int deviceID);


#endif
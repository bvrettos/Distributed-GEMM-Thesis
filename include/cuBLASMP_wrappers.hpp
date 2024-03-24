#ifndef CUBLASMP_WRAPPERS_HPP
#define CUBLASMP_WRAPPERS_HPP

#include <iostream>
#include "calSetup.hpp"
#include "errorHandling.hpp"
#include "cmatrix.h"
#include "pblasDecomposition.hpp"

void cuBLASMpDgemmWrap(char TransA,  char TransB, long int M, long int N, long int K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC, int Mb, int Nb, int dRow, int dCol);

#endif
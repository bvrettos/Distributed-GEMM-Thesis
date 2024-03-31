#ifndef CUBLASMP_WRAPPERS_HPP
#define CUBLASMP_WRAPPERS_HPP

#include <iostream>
#include "cudaCommunicator.hpp"
#include "calSetup.hpp"
#include "errorHandling.hpp"
#include "cmatrix.h"
#include "pblasDecomposition.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <vector>
#include <cublasmp.h>
#include <validation.hpp>
#include <cblas.h>
#include <unistd.h>
#include <logging.hpp>

void cuBLASMpDgemmWrap(char TransA,  char TransB, long int M, long int N, long int K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC, long int Mb, long int Nb, int dRow, int dCol);

#endif
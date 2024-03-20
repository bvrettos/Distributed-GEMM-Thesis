#ifndef CUBLASMP_WRAPPERS_HPP
#define CUBLASMP_WRAPPERS_HPP

#include <iostream>
#include "calSetup.hpp"

void cuBLASMpDgemmWrap(char TransA,  char TransB, long int M, long int N, long int K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC, int dev_ids[]);

void cuBLASMpSgemmWrap(char TransA,  char TransB, long int M, long int N, long int K,
  float alpha, float* A, long int ldA, float* B, long int ldB, float beta, float* C,
  long int ldC, int dev_ids[]);

#endif
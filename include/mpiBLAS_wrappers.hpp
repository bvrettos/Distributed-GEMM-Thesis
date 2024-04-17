#ifndef MPI_BLAS_WRAPPERS_HPP
#define MPI_BLAS_WRAPPERS_HPP

#include <PARALiA.hpp>
#include "2DBlockSequentialDecomposition.hpp"
#include "2DBlockCyclicDecomposition.hpp"
#include "cudaCommunicator.hpp"
#include <cblas.h>
#include <logging.hpp>

double PARALiA_MPI_Dgemm(char TransA,  char TransB, const long long M, const long long N, const long long K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC);

double MPI_Dgemm_Sequential(char TransA,  char TransB, long int M, long int N, long int K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC);

double MPI_Dgemm_Cyclic(char TransA,  char TransB, long int M, long int N, long int K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC, int blockRows, int blockColumns);

#endif
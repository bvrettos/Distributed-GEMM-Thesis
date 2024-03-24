#ifndef MPI_BLAS_WRAPPERS_HPP
#define MPI_BLAS_WRAPPERS_HPP

#include <PARALiA.hpp>
#include "2DBlockSequentialDecomposition.hpp"
#include "2DBlockCyclicDecomposition.hpp"
#include <cblas.h>

double PARALiA_MPI_Dgemm_Controlled(char TransA,  char TransB, long int M, long int N, long int K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC, int dev_ids[]);

double PARALiA_MPI_Dgemm(char TransA,  char TransB, long int M, long int N, long int K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC);

double MPI_Dgemm_Sequential(char TransA,  char TransB, long int M, long int N, long int K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC);

double MPI_Dgemm_Cyclic(char TransA,  char TransB, long int M, long int N, long int K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC, int blockRows, int blockColumns);

#endif
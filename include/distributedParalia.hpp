#ifndef MPI_BLAS_WRAPPERS_HPP
#define MPI_BLAS_WRAPPERS_HPP

#include <PARALiA.hpp>
#include "2DBlockSequentialDecomposition.hpp"
#include "2DBlockCyclicDecomposition.hpp"
#include "cudaCommunicator.hpp"
#include <cblas.h>
#include <logging.hpp>
#include <cmatrix.h>

void preDistributedSequentialParaliaGemm(char TransA,  char TransB, const long long M, const long long N, const long long K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC, int numberOfRuns, bool logging, bool gatherResults, int aLoc, int bLoc, int cLoc);

void validateDistributedSequentialParalia(char TransA,  char TransB, const long long M, const long long N, const long long K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC);

void paraliaFullGemmOffloadSequential(char TransA,  char TransB, const long long M, const long long N, const long long K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC, int numberOfRuns, bool logging, bool warmup, int aLoc, int bLoc, int cLoc);

#endif
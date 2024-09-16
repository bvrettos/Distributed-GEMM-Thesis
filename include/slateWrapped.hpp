#ifndef SLATE_WRAPPED_HPP
#define SLATE_WRAPPED_HPP

#include <slate/slate.hh>
#include <blas.hh>
#include <cblas.h>
#include <stdio.h>
#include <math.h>
#include <cudaCommunicator.hpp>
#include <cmatrix.h>
#include <validation.hpp>
#include <logging.hpp>
#include <generalUtilities.hpp>
#include <iostream>

template <typename matrix_type>
void random_matrix(matrix_type& A);

template <typename scalar_type>
void validateGEMM(char TransA,  char TransB, const long long M, const long long N, const long long K,
  scalar_type alpha, scalar_type* A, long long int ldA, scalar_type* B, long long int ldB, scalar_type beta, scalar_type* C, long long int ldC
  ,long mb, long nb);

template <typename scalar_type>
void slateGEMM(char TransA,  char TransB, const long long M, const long long N, const long long K,
  scalar_type alpha, scalar_type* A, long long int ldA, scalar_type* B, long long int ldB, scalar_type beta, scalar_type* C, long long int ldC,
  long mb, long nb, bool logging, bool gatherResults, int initialDataLocation);

template <typename scalar_type>
void slatePreDistributedGemm(char TransA,  char TransB, const long long M, const long long N, const long long K,
  scalar_type alpha, scalar_type* A, long long int ldA, scalar_type* B, long long int ldB, scalar_type beta, scalar_type* C, long long int ldC
  ,long mb, long nb, int numberOfRuns, bool logging, bool gatherResults, int initialDataLocation);

template <typename scalar_type>
void slateFullGemmOffload(char TransA,  char TransB, const long long M, const long long N, const long long K,
  scalar_type alpha, scalar_type* A, long long int ldA, scalar_type* B, long long int ldB, scalar_type beta, scalar_type* C, long long int ldC
  ,long mb, long nb,int numberOfRuns, bool logging, bool gatherResults, int initialDataLocation);

#endif
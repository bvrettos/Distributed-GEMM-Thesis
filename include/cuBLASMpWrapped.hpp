#ifndef CUBLASMP_WRAPPERS_HPP
#define CUBLASMP_WRAPPERS_HPP

#include <iostream>
#include "cudaCommunicator.hpp"
#include "calSetup.hpp"
#include "errorHandling.hpp"
#include "cmatrix.h"
#include "pblasDecomposition.hpp"

#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <vector>
#include <cublasmp.h>
#include <validation.hpp>
#include <cblas.h>
#include <unistd.h>
#include <logging.hpp>
#include <generalUtilities.hpp>

template <typename scalar_t>
void cuBLASMpGEMMWrap(char TransA,  char TransB, const long long M, const long long N, const long long K,
  scalar_t alpha, scalar_t* A, const long long ldA, scalar_t* B, const long long ldB, scalar_t beta, scalar_t* C,
  const long long ldC, long int Mb, long int Nb);

template <typename scalar_t>
void cuBLASMpPreDistributedGemm(char TransA,  char TransB, const long long M, const long long N, const long long K,
  scalar_t alpha, scalar_t* A, const long long ldA, scalar_t* B, const long long ldB, scalar_t beta, scalar_t* C,
  const long long ldC, long int Mb, long int Nb, int numberOfRuns, bool logging, bool gatherResults);

template <typename scalar_t>
void cuBLASMpFullGemmOffload(char TransA,  char TransB, const long long M, const long long N, const long long K,
  scalar_t alpha, scalar_t* A, const long long ldA, scalar_t* B, const long long ldB, scalar_t beta, scalar_t* C,
  const long long ldC, long int Mb, long int Nb, int numberOfRuns, bool logging, bool gatherResults);

#endif
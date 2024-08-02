#ifndef COSMA_WRAPPED_HPP
#define COSMA_WRAPPED_HPP

#include <iostream>
#include <algorithm>
#include <cstdio>
#include <cmatrix.h>
#include <errorHandling.hpp>
#include <generalUtilities.hpp>
#include <logging.hpp>
#include <costa/layout.hpp>
#include <pblasDecomposition.hpp>
#include <cosma/multiply.hpp>
#include <cblas.h>
#include <validation.hpp>

template <typename scalar_t>
void cosmaGEMM(char TransA, char TransB, const long long M, const long long N, const long long K, scalar_t alpha, scalar_t* A, const long long lda,
    scalar_t* B, const long long ldb, scalar_t beta, scalar_t* C, const long long ldc, MPI_Comm comm);

template <typename scalar_t>
void validateCosmaGEMM(char TransA, char TransB, const long long M, const long long N, const long long K, scalar_t alpha, scalar_t* A, const long long lda,
    scalar_t* B, const long long ldb, scalar_t beta, scalar_t* C, const long long ldc, MPI_Comm comm);

template <typename scalar_t>
void cosmaPreDistributedOptimalGemm(char TransA, char TransB, const long long M, const long long N, const long long K, scalar_t alpha, scalar_t* A, const long long lda,
    scalar_t* B, const long long ldb, scalar_t beta, scalar_t* C, const long long ldc, MPI_Comm comm, const int numberOfRuns, bool logging);

template <typename scalar_t>
void cosmaPreDistributedParaliaGemm(char TransA, char TransB, const long long M, const long long N, const long long K, scalar_t alpha, scalar_t* A, const long long lda,
    scalar_t* B, const long long ldb, scalar_t beta, scalar_t* C, const long long ldc, MPI_Comm comm, const int numberOfRuns, bool logging);

template <typename scalar_t>
void cosmaPreDistributedScalapackGemm(char TransA, char TransB, const long long M, const long long N, const long long K, scalar_t alpha, scalar_t* A, const long long lda,
    scalar_t* B, const long long ldb, scalar_t beta, scalar_t* C, const long long ldc, const int Mb, const int Nb, MPI_Comm comm, const int numberOfRuns, bool logging);

template <typename scalar_t>
void cosmaFullGemmOffloadScalapack(char TransA, char TransB, const long long M, const long long N, const long long K, scalar_t alpha, scalar_t* A, const long long lda,
    scalar_t* B, const long long ldb, scalar_t beta, scalar_t* C, const long long ldc, int Mb, int Nb, MPI_Comm comm, const int numberOfRuns, bool logging, bool gatherResult);

template <typename scalar_t>
void cosmaFullGemmOffloadParalia(char TransA, char TransB, const long long M, const long long N, const long long K, scalar_t alpha, scalar_t* A, const long long lda,
    scalar_t* B, const long long ldb, scalar_t beta, scalar_t* C, const long long ldc, MPI_Comm comm, const int numberOfRuns, bool logging);

#endif
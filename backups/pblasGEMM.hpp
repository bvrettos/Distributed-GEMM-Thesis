#ifndef PBLAS_GEMM_HPP
#define PBLAS_GEMM_HPP

#include <pblasDecomposition.hpp>
#include <validation.hpp>
#include <cblas.h>

extern "C" {
  /* Cblacs declarations */
  void Cblacs_pinfo(int*, int*);
  void Cblacs_get(int, int, int*);
  void Cblacs_gridinit(int*, const char*, int, int);
  void Cblacs_pcoord(int, int, int*, int*);
  void Cblacs_gridexit(int);
  void Cblacs_barrier(int, const char*);
 
  int numroc_(int*, int*, int*, int*, int*);

  void descinit_(int *desc, const int *m,  const int *n, const int *mb, 
    const int *nb, const int *irsrc, const int *icsrc, const int *ictxt, 
    const int *lld, int *info);

  void pdgemm_( char* TRANSA, char* TRANSB,
                int * M, int * N, int * K,
                double * ALPHA,
                double * A, int * IA, int * JA, int * DESCA,
                double * B, int * IB, int * JB, int * DESCB,
                double * BETA,
                double * C, int * IC, int * JC, int * DESCC );

}

void pblasDgemm(char* TransA, char* TransB, int M, int N, int K, double alpha, double* A, int lda, double* B, int ldb, double beta, double* C, int ldc, int Mb, int Nb, int dRow, int dCol);
// void scalapackGemm(char* TransA, char* TransB, int M, int N, int K, double alpha, double* A, int lda, double* B, int ldb, double beta,double* C, int ldc, int Mb, int Nb, int dRow, int dCol);

#endif
#ifndef SUMMA_HPP
#define SUMMA_HPP

#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "errorHandling.hpp"
#include <map>
#include <tuple>
#include <iostream>
#include <cmatrix.h>
#include <transfers.hpp>
#include <cublas_v2.h>
#include <validation.hpp>
#include <cblas.h>
#include <logging.hpp>
#include <nccl.h>


class Summa25Decomposer {
    private:
        void createCommunicators();
        void reduceResult();
        void broadcastToStack();
        void initializeNCCL();
        
    public:
        long long M, N, K; /* M = N = K */
        int dRow, dCol, c;  /* dRow = dCol = p */
        int rank, size; /* MPI_COMM_WORLD */
        MPI_Comm commonStackCommunicator, commonGridCommunicator;
        int processRow, processColumn, processStack; /* i, j, k*/
        int pc3, pc;
        int commonGridRank, commonStackRank;
        bool broadcastComplete;
        
        /* 
            Local pointers are where the GEMM operations will take place. 
            These can go either in host memory or device memory (device is better). 
            Allocation happens inside of this class scope (for now, needed for overlap).
        */

        double *localA, *localB, *localC; /* Memory where normal execution takes place, not enough if overlap needs to happen */
        long long localM, localN, localK; 
        long long llda, lldb, lldc; // Not really necessary 

        /* Communication/Computation overlap */
        cublasHandle_t cublasContext;
        cudaStream_t *streams;
        int activeStreams;
        ncclUniqueId uniqueId;
        ncllComm_t ncclCommunicator, ncclCommonStackCommunicator;
    
        Summa25Decomposer(long long M, long long N, long long K, int dRow, int dCol, int c);
        void multiply(char TransA, char TransB, double* A, double* B, double* C, double alpha, double beta);
        void decompose2D(int senderRank, double* A, double* B, double* C);
        void reorderMatrix(int gatherRank, double* C);
        ~Summa25Decomposer();
};

void preDistributedSumma25Dgemm(char TransA, char TransB, long long M, long long N, long long K, 
    double alpha, double* A, long long lda, double* B, long long ldb, double beta, double* C, long long ldc, int dRow, int dCol, int c,
    int numberOfRuns, bool logging, bool gatherResults);

void fullOffloadSumma25Dgemm(char TransA, char TransB, long long M, long long N, long long K, 
    double alpha, double* A, long long lda, double* B, long long ldb, double beta, double* C, long long ldc, int dRow, int dCol, int c,
    int numberOfRuns, bool logging, bool gatherResults);

#endif
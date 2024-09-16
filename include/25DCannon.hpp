#ifndef CANNON_HPP
#define CANNON_HPP

/* Standard Stuff */
#include <mpi.h>
#include <cblas.h>

/* Custom libraries and helpers */
#include "errorHandling.hpp"
#include <cmatrix.h>
#include <transfers.hpp>
#include <validation.hpp>
#include <logging.hpp>

/* Standard C/C++ */
#include <map>
#include <tuple>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstdio>

/* CUDA Related */
#include <nccl.h>
#include <cuda.h>
#include <cublas_v2.h>

int truncatedDivisionRemainer(int a, int b);

class Summa25Decomposer {
    private:
        /* MPI Version */
        void createCommunicators();
        void reduceResult(double* cTile);
        void broadcastToStack();

        /* NCCL Version */
        void initializeNCCL();
        void communicationComputationOverlapInit();
        void broadcastToStackNCCL(double* aTile, double* bTile);
        void reduceResultNCCL(double* cTile);
        void scatterDecompose(int senderRank, double *A, double *B, double *localA, double *localB);
        void gatherMatrix(int gatherRank, double *C, double *cTile);
        

    public:
        long long M, N, K; /* M = N = K */
        int dRow, dCol, c;  /* dRow = dCol = p */
        int rank, size; /* MPI_COMM_WORLD */

        MPI_Comm commonStackCommunicator, commonGridCommunicator;
        int processRow, processColumn, processStack; /* i, j, k*/
        int pc3, pc;
        int commonGridRank, commonStackRank;
        bool broadcastComplete; 
        cublasHandle_t cublasContext;
        
        /* 
            Local pointers are where the GEMM operations will take place. 
            These can go either in host memory or device memory (device is better). 
            Allocation happens inside of this class scope (for now, needed for overlap).
        */

        double *localA, *localB, *localC; /* Memory where normal execution takes place, not enough if overlap needs to happen */
        long long localM, localN, localK; 
        long long llda, lldb, lldc; // Not really necessary 
        double communicationTime, executionTime;   // Metric Counters

        /* Communication/Computation overlap */
        bool communicationComputationOverlap;
        cudaStream_t *streams;  /* Streams used for NCCL communication and memory copying */
        cudaEvent_t *events; /* Events used for synchronizing NCCL communication and cuBLAS computation */
        cudaStream_t cublasStream; /* Main computation stream since all cuBLAS calls are serialized */

        int activeStreams;
        ncclUniqueId uniqueId;
        ncclComm_t ncclCommunicator, ncclCommonStackCommunicator;
        double **workspaceA, **workspaceB;
    
        Summa25Decomposer(long long M, long long N, long long K, int dRow, int dCol, int c, bool communicationComputationOverlap);
        /* First one is naive MPI version, second is NCCL with communication/computation overlap */
        void multiply(char TransA, char TransB, double* A, double* B, double* C, double alpha, double beta);
        void multiplyNCCL(char TransA, char TransB, double* A, double* B, double* C, double alpha, double beta);
        void serializedMultiplyNCCL(char TransA, char TransB, double* A, double* B, double* C, double alpha, double beta);

        /* These are common for both versions*/
        void decompose2D(int senderRank, double* A, double* B, double* C, double* aTile, double* bTile, double* cTile);
        void reorderMatrix(int gatherRank, double* C, double* cTile);
        void resetMetricCounters();

        ~Summa25Decomposer();
};

void preDistributedSumma25Dgemm(char TransA, char TransB, long long M, long long N, long long K, 
    double alpha, double* A, long long lda, double* B, long long ldb, double beta, double* C, long long ldc, int dRow, int dCol, int c,
    int numberOfRuns, bool logging, bool gatherResults);

void fullOffloadSumma25Dgemm(char TransA, char TransB, long long M, long long N, long long K, 
    double alpha, double* A, long long lda, double* B, long long ldb, double beta, double* C, long long ldc, int dRow, int dCol, int c,
    int numberOfRuns, bool logging, bool gatherResults);

void preDistributedCannon25NCCL(char TransA, char TransB, long long M, long long N, long long K, 
    double alpha, double* A, long long lda, double* B, long long ldb, double beta, double* C, long long ldc, int dRow, int dCol, int c,
    int numberOfRuns, bool logging, bool gatherResults);

#endif
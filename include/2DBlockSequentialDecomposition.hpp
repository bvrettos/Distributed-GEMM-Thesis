#ifndef BLOCK_SEQUENTIAL_DECOMPOSITION_HPP
#define BLOCK_SEQUENTIAL_DECOMPOSITION_HPP

#include <mpi.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include "errorHandling.hpp"

class GEMM_BlockSequentialDecomposer {
    private:
    public:
        /* GEMM and Grid specifications */
        int M, N, K;
        int localM, localN, localK;
        int dRow, dCol, numberOfDevices;

        /* Process Values */
        int rank;
        int communicatorSize;
        
        /* MPI Related Structures */
        MPI_Comm GEMM_Communicator;
        MPI_Datatype localBlockA, localBlockB, localBlockC;
        MPI_Datatype globalBlockA, globalBlockB, globalBlockC;
        MPI_Datatype dummy;

        /* Scatter Values */
        int *scatterOffsetA, *scatterOffsetB, *scatterOffsetC;
        int *scatterCountA, *scatterCountB, *scatterCountC;

        GEMM_BlockSequentialDecomposer(int M, int N, int K, MPI_Comm communicator);

        void calculateVirtualDeviceGrid();
        void allocateMPIDatatypes();
        void calculateScatterValues();
        void scatterMatrices(double* A, double* B, double* C, double* localA, double* localB, double* localC);
        void gatherResult(double* C, double* localC);

        ~GEMM_BlockSequentialDecomposer();
};

#endif
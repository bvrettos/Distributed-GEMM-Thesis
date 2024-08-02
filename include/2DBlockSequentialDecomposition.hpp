#ifndef BLOCK_SEQUENTIAL_DECOMPOSITION_HPP
#define BLOCK_SEQUENTIAL_DECOMPOSITION_HPP

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

class BlockSequentialDecomposer {
    private:
    public:
        /* GEMM and Grid specifications */
        int M, N, K;
        int localM, localN, localK;
        int dRow, dCol, numberOfDevices;
        bool colMajor;

        /* Process Values */
        int rank;
        int size;
        int processRow, processColumn;
        
        /* MPI Related Structures */
        MPI_Comm GEMM_Communicator;

        /* Scatter Values */
        BlockSequentialDecomposer(int M, int N, int K, MPI_Comm communicator, bool colMajor=true);
        void calculateVirtualDeviceGrid();
        void deliverMatrix(double* globalA, double* globalB, double* globalC, double** localA, double** localB, double** localC);
        void gatherResult(int gatherRank, double* C, double* localC);

        ~BlockSequentialDecomposer();
};

#endif
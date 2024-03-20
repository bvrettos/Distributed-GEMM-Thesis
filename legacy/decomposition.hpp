#ifndef DECOMPOSITION_HPP
#define DECOMPOSITION_HPP

#include <mpi.h>
#include <cuda_runtime.h>
#include "errorHandling.hpp"

#include <iostream>
#include <cmath>
#include <map>
#include <tuple>
#include <vector>

class blockSequentialDecomposer {
    public:
        /* Datatypes needed to scatter matrices */
        MPI_Datatype blockA, blockB, blockC, dummy;
        MPI_Datatype globalA, globalB, globalC;

        /* Communicator used for scattering/gathering.
            Can be used for choosing execution device */
        const MPI_Comm communicator;

        /* Scatter arrays (variemai na eksigw twra :( )*/
        int *scatterOffsetA, *scatterOffsetB, *scatterOffsetC;
        int *scatterCountA, *scatterCountB, *scatterCountC;

        /* GEMM Problem */
        int M, N, K;
        int dRow, dCol, numDevices;
        int localM, localN, localK;

        blockSequentialDecomposer(const int M, const int N, const int K, MPI_Comm communicator);
        ~blockSequentialDecomposer();

        void calculateGridDimensions();
        void allocateMPIDatatypes();
        void calculateScatterValues();
        void scatterMatrices(double* A, double* B, double* C, double* localA, double* localB, double* localC);
        void getDecompositionValues(int* localM, int* localN, int* localK);
        void gatherResult(double* C, double* localC);
};

#endif
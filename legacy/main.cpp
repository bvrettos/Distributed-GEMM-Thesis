#include <mpi.h>
#include <stdio.h>
#include <stdlib.h> // rand()

#include <cblas.h>

#include "matrix.hpp"
#include <iostream>
#include <vector>
#include <string>

#include "gemm.hpp"
#include "cmatrix.h"

#define VALIDATE

int main(int argc, char* argv[]) {
    // Initialize the MPI environment
    MPI_Init(&argc, &argv);

    /* Get rank, size and output it for debugging reasons */
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 3)
    {
        std::cerr << "Usage: mpirun -np ... ./main dRow dCol" << std::endl;
        exit(-1);
    }

    int dRow, dCol;
    dRow = atoi(argv[1]); //Py
    dCol = atoi(argv[2]); //Px

    if (dRow*dCol != size) {
        std::cerr << "dRow*dCol must be equal to number of processes" << std::endl;
        exit(-2);
    }

    /* GEMM information */
    int m,n,k;
    m = 400;
    n = 100;
    k = 2048;

    double alpha, beta;
    alpha = 1.0;
    beta = 1.0;

    double *globalA, *globalB, *globalC, *referenceC;

    /* Global matrices */
    if (rank == 0) {
        std::cout << "World Size: " << size << " Process Rank: " << rank << std::endl;
        globalA = (double *) malloc(sizeof(double)*m*k);
        globalB = (double *) malloc(sizeof(double)*k*n);
        globalC = (double *) malloc(sizeof(double)*m*n);

        generateMatrix(globalA, m, k);
        generateMatrix(globalB, k, n);
        generateMatrix(globalC, m, n);

        referenceC = copyMatrix(globalC, m, n);
    }

    int localM, localN, localK;

    /* Start decomposition from C - Ignore K dimension */
    localK = k;
    localM = m/dRow;
    localN = n/dCol;
    
    /* Initialize and allocate local Matrices */
    double *localA, *localB, *localC;
    localA = (double *) malloc(sizeof(double)*localM*localK);
    localB = (double *) malloc(sizeof(double)*localK*localN);
    localC = (double *) malloc(sizeof(double)*localM*localN);

    /* Local 2D block MPI Definitions */
    MPI_Datatype blockA, blockB, blockC, dummy;
    MPI_Type_vector(localM, localK, localK, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &blockA);
    MPI_Type_commit(&blockA);

    MPI_Type_vector(localK, localN, localN, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &blockB);
    MPI_Type_commit(&blockB);

    MPI_Type_vector(localM, localN, localN, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &blockC);
    MPI_Type_commit(&blockC);
    

    /* Global 2D block MPI Definitions */
    MPI_Datatype globalBlockA, globalBlockB, globalBlockC;
    MPI_Type_vector(localM, localK, k, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &globalBlockA);
    MPI_Type_commit(&globalBlockA);

    MPI_Type_vector(localK, localN, n, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &globalBlockB);
    MPI_Type_commit(&globalBlockB);

    MPI_Type_vector(localM, localN, n, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &globalBlockC);
    MPI_Type_commit(&globalBlockC);

    int *scatterOffsetA, *scatterCountA;
    int *scatterOffsetB, *scatterCountB;
    int *scatterOffsetC, *scatterCountC;

    if (rank == 0)
    {
        std::cout << "Block M: " << localM << " Block N: " << localN << " Block K: " << localK << std::endl;
        scatterOffsetA = (int*) malloc(size*sizeof(int));
        scatterCountA = (int*) malloc(size*sizeof(int));

        scatterOffsetB = (int*) malloc(size*sizeof(int));
        scatterCountB = (int*) malloc(size*sizeof(int));

        scatterOffsetC = (int*) malloc(size*sizeof(int));
        scatterCountC = (int*) malloc(size*sizeof(int));

        /* C Scattering 2D Sequential Decomp*/
        for (int i = 0; i < dRow; i++) {   
            for (int j = 0; j < dCol; j++) {
                scatterCountC[i*dCol + j] = 1;
                scatterOffsetC[i*dCol + j] = localM * localN * dCol * i + localN * j;
            }
        }

        /* A Scattering 1D Decomp*/
        for (int i = 0; i < dRow; i++) {
            for (int j = 0; j < dCol; j++) {
                scatterCountA[i*dCol + j] = 1;
                scatterOffsetA[i*dCol + j] = localM*localK*i;
            }
        }

        /* B Scattering 1D Decomp*/
        for (int i = 0; i < dCol; i++) {
            for (int j = 0; j < dRow; j++) {
                scatterCountB[j*dCol + i] = 1;
                scatterOffsetB[j*dCol + i] = localN*i;
            }
        }

        #ifdef DEBUG
            for (int k = 0; k < size; k++) {
                std::cout << "ScatteroffsetA: " << scatterOffsetA[k] << std::endl;
                std::cout << "ScatteroffsetB: " << scatterOffsetB[k] << std::endl;
                std::cout << "ScatteroffsetC: " << scatterOffsetC[k] << std::endl;
                std::cout << std::endl;
            }
        #endif
    }

    int ierr = MPI_Scatterv(globalA, scatterCountA, scatterOffsetA, globalBlockA, localA, 1, blockA, 0, MPI_COMM_WORLD);
    if (ierr != MPI_SUCCESS) {
        std::cerr << "Help" << std::endl;
    }

    ierr = MPI_Scatterv(globalC, scatterCountC, scatterOffsetC, globalBlockC, localC, 1, blockC, 0, MPI_COMM_WORLD);
    if (ierr != MPI_SUCCESS) {
        std::cerr << "Help" << std::endl;
    }

    ierr = MPI_Scatterv(globalB, scatterCountB, scatterOffsetB, globalBlockB, localB, 1, blockB, 0, MPI_COMM_WORLD);
    if (ierr != MPI_SUCCESS) {
        std::cerr << "Help" << std::endl;
    }

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, localM, localN, localK, alpha, localA,
            localK, localB, localN, beta, localC, localN);

    MPI_Gatherv(localC, 1, blockC, globalC, scatterCountC, scatterOffsetC, globalBlockC, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        Matrix<double> mpiC(m, n, globalC);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, alpha, globalA,
            k, globalB, n, beta, referenceC, n);

        Matrix<double> refC(m, n , referenceC);

        #ifdef VALIDATE
            refC.writeMatrix("referenceC.txt");
            mpiC.writeMatrix("mpiC.txt");
            if (mpiC == refC) {
                std::cout << "Matrix validation passed" << std::endl;
            }
            else
                std::cout << "Matrix validation failed" << std::endl;
        #endif

        free(globalA);
        free(globalB);
        free(globalC);
    }

    free(localA);
    free(localB);
    free(localC);

    // Finalize the MPI environment.
    MPI_Finalize();
}
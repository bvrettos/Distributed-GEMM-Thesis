#include "mpiBLAS_wrappers.hpp"

double MPI_Dgemm_Sequential(char TransA,  char TransB, long int M, long int N, long int K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC)
{
    CBLAS_TRANSPOSE gemmTransA, gemmTransB;
    gemmTransA = CblasNoTrans;
    gemmTransB = CblasNoTrans;

    if (TransA == 'n' || TransA == 'N') {
        gemmTransA = CblasNoTrans;
    }
    else if (TransA == 't' || TransA == 'T' || TransA == 'c' || TransA == 'C') {
        gemmTransA = CblasTrans;
    }
    else {
        /* Translation not recognized, return error*/
        return -1;
    }

    if (TransB == 'n' || TransB == 'N') {
        gemmTransB = CblasNoTrans;
    }
    else if (TransB == 't' || TransB == 'T' || TransB == 'c' || TransB == 'C') {
        gemmTransB = CblasTrans;
    }
    else {
        /* Translation not recognized, return error*/
        return -1;
    }

    /* Check if MPI has been initialized */
    int initializedMPI;
    MPI_Initialized(&initializedMPI);

    if (!initializedMPI) {
        std::cerr << "ERROR: MPI has not been initialized. Call MPI_Init before calling this function" << std::endl;
        exit(-2);
    }

    MPI_Comm problemCommunicator = MPI_COMM_WORLD;

    GEMM_BlockSequentialDecomposer Decomposer(M, N, K, problemCommunicator);

    int rank = Decomposer.rank;
    
    double *localA, *localB, *localC;

    int localM = Decomposer.localM;
    int localN = Decomposer.localN;
    int localK = Decomposer.localK;

    int llda = localK;
    int lldb = localN;
    int lldc = localN;

    double t1 = MPI_Wtime();

    localA = (double*) malloc(sizeof(double) * localM * localK);
    localB = (double*) malloc(sizeof(double) * localN * localK);
    localC = (double*) malloc(sizeof(double) * localM * localN);

    Decomposer.scatterMatrices(A, B, C, localA, localB, localC);
    double t2 = MPI_Wtime();

    cblas_dgemm(CblasRowMajor, gemmTransA, gemmTransB, localM, localN, localK, alpha, localA, llda, localB, lldb, beta, localC, lldc);

    double t3 = MPI_Wtime();

    Decomposer.gatherResult(C, localC);

    free(localA);
    free(localB);
    free(localC);

    return t3-t2;
}

double MPI_Dgemm_Cyclic(char TransA,  char TransB, long int M, long int N, long int K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC, int blockRows, int blockColumns)
{
    // /* Check if MPI has been initialized */
    // int initializedMPI;
    // MPI_Initialized(&initializedMPI);

    // if (!initializedMPI) {
    //     std::cerr << "ERROR: MPI has not been initialized. Call MPI_Init before calling this function" << std::endl;
    //     exit(-2);
    // }

    // MPI_Comm problemCommunicator = MPI_COMM_WORLD;

    // double timer1, timer2, timer3, timer4, timer5;
    // timer1 = MPI_Wtime();
    // GEMM_BlockCyclicDecomposer decomposer(M, N, K, blockRows, blockColumns, problemCommunicator);

    // decomposer.calculateTaskMap();

    // timer2 = MPI_Wtime();

    // int tilesPerTask = decomposer.helperTilesPerTask;
    // int cTilesPerDevice = decomposer.cTilesPerDevice;

    // /* Now, each device is supposed to iterate their own task map and allocate memory depending on what info needs to be sendedn to them. */

    // /* 
    //     Prepeare a set for A and B. Every time you want to send an tile from either A or B, check if the index is in the set. If not, then send and update the set.
    //     If you find the index, then skip the send. This can be done in a static manner (I think).
    // */
    // /* Do not allocate further memory since the last dimension is quite more intensive. Do that only if the task needs to allocate memory */
    // /* Going full r-word mode... */

    // /* Prepare memory for each task */
    // double** localC;             // localC[cTilesPerDevice][blockRow*blockColumns];
    // double ***localA, ***localB; // localA[cTilesPerDevice][tilesPerTask][blockRow * blockColumns]
    
    // timer3 = MPI_Wtime();

    // localA = (double***) malloc(sizeof(double**) * cTilesPerDevice);
    // localB = (double***) malloc(sizeof(double**) * cTilesPerDevice);
    // localC = (double**) malloc(sizeof(double*) * cTilesPerDevice);

    // for (int i = 0; i < cTilesPerDevice; i++) {
    //     localC[i] = (double*) malloc(sizeof(double) * blockRows * blockColumns);
    //     localA[i] = (double**) malloc(sizeof(double*) * tilesPerTask);
    //     localB[i] = (double**) malloc(sizeof(double*) * tilesPerTask);
    //     for (int j = 0; j < tilesPerTask; j++) {
    //         localA[i][j] = (double*) malloc(sizeof(double) * blockRows * blockColumns);
    //         localB[i][j] = (double*) malloc(sizeof(double) * blockRows * blockColumns);
    //     }
    // }

    // timer4 = MPI_Wtime();

    // if (decomposer.squareDecomposition)
    //     decomposer.squareTaskScattering(A, B, C, localA, localB, localC);
    // else {
    //     std::cout << "Non square cyclic decomposition not ready yet, sorry :( " << std::endl;
    //     return 0;
    // }

    // timer5 = MPI_Wtime();

    // double decomposerTime = timer2-timer1;
    // double scatteringTime = timer5-timer4;
    // double allocationTime = timer4-timer3;

    // /* Actually run DGEMM */
    // double t1, t2;
    // t1 = MPI_Wtime();
    
    // for (int i = 0; i < cTilesPerDevice; i++) {
    //     cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, blockRows, blockRows, blockRows,
    //      alpha, localA[i][0], blockRows, localB[i][0], blockRows, beta, localC[i], blockRows);
    //     for (int j = 1; j < tilesPerTask; j++) {
    //         cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, blockRows, blockRows, blockRows,
    //         alpha, localA[i][j], blockRows, localB[i][j], blockRows, 1.0000000, localC[i], blockRows);
    //     }
    // }

    // t2 = MPI_Wtime();

    // /* Gather */
    // decomposer.squareTaskGathering(C, localC);
    // if (decomposer.rank == 0)
    //     printf("Decomposer Time: %lf Scattering Time: %lf Allocation Time: %lf Execution Time: %lf\n", decomposerTime, scatteringTime, allocationTime, t2-t1);

    // MPI_Barrier(problemCommunicator);

    // return t2-t1;
}
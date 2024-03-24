#include <mpi.h>
#include "mpiBLAS_wrappers.hpp"
#include "cmatrix.h"
#include "validation.hpp"

// #define VALIDATE
// #define DEBUG

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* GEMM Problem Values */
    int M, N, K;
    double alpha, beta;

    M = 8192;
    N = 8192;
    K = 8192;
    
    alpha = 0.1234;
    beta = 1.2313;

    double *A, *B, *C;
    double *referenceC;
    if (rank == 0) {
        /* Generate matrices */
        A = (double*) malloc(sizeof(double)*M*K);
        B = (double*) malloc(sizeof(double)*K*N);
        C = (double*) malloc(sizeof(double)*M*N);

        generateMatrix(A, M, K);
        generateMatrix(B, K, N);
        generateMatrix(C, M, N);
    #ifdef DEBUG
        printMatrix(A, M, K, rank);
        printMatrix(B, N, K, rank);
        printMatrix(C, M, N, rank);
    #endif
    
    #ifdef VALIDATE
        referenceC = copyMatrix(C, M, N);
    #endif
    }

    int blockRows = 1024;
    int blockColumns = 1024;
    double t1, t2;
    double computationTime;
    t1 = MPI_Wtime();
    // computationTime = MPI_Dgemm_Sequential('N', 'N', M, N, K, alpha, A, K, B, N, beta, C, N);

    computationTime = MPI_Dgemm_Cyclic('N', 'N', M, N, K, alpha, A, K, B, N, beta, C, N, blockRows, blockColumns);
    // PARALiA_MPI_Dgemm('N', 'N', M, N, K, alpha, A, K, B, N, beta, C, N);
    t2 = MPI_Wtime();

    if (rank == 0){
        printf("Elapsed time is %lf. Computation time is %lf\n", t2-t1, computationTime);
    }

    if (rank == 0) {
        #ifdef VALIDATE
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, referenceC, N);
            Dtest_equality(C, referenceC, M*N);
        #endif
    }

    MPI_Finalize();

    return 0;
}
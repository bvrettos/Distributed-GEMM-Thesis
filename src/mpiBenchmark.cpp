#include <mpi.h>
#include "mpiBLAS_wrappers.hpp"
#include "cmatrix.h"
#include "validation.hpp"

// #define VALIDATE

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* GEMM Problem Values */
    int M, N, K;
    double alpha, beta;

    M = 100;
    N = 100;
    K = 100;
    
    alpha = 0.1234;
    beta = 1.231;

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
    
    #ifdef VALIDATE
        referenceC = copyMatrix(C, M, N);
    #endif
    }

    // MPI_Dgemm_Sequential('N', 'N', M, N, K, alpha, A, K, B, N, beta, C, N);
    MPI_Dgemm_Cyclic('N', 'N', M, N, K, alpha, A, K, B, N, beta, C, N);
    //PARALiA_MPI_Dgemm('N', 'N', M, N, K, alpha, A, K, B, N, beta, C, N);

    if (rank == 0) {
        #ifdef VALIDATE
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, referenceC, N);
            Dtest_equality(C, referenceC, M*N);
        #endif
    }

    MPI_Finalize();

    return 0;
}
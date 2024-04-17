#include <mpi.h>
#include "mpiBLAS_wrappers.hpp"
#include "cmatrix.h"
#include "validation.hpp"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* GEMM Problem Values */
    int M, N, K;
    double alpha, beta;
    if (argc < 7) {
        printf("Usage: ./mpiBenchmark 'NN' M N K alpha beta");
        exit(1);
    }
    char transposeA = argv[1][0];
    char transposeB = argv[1][1];
    M = atoi(argv[2]);
    N = atoi(argv[3]);
    K = atoi(argv[4]);
    
    alpha = atof(argv[5]);
    beta = atof(argv[6]);

    double *A, *B, *C;
    double *referenceC;
    
    if (rank == 0) {
        A = (double*) malloc(sizeof(double) * M * K);
        B = (double*) malloc(sizeof(double) * N * K);
        C = (double*) malloc(sizeof(double) * M * N);
        MatrixInit(A, M, K, -1);
        MatrixInit(B, K, N, -1);
        MatrixInit(C, M, N, -1);    
        #ifdef VALIDATE
            referenceC = copyMatrix(C, M, N);
        #endif
    }

    double t1, t2;
    double computationTime;
    
    t1 = MPI_Wtime();

    computationTime = MPI_Dgemm_Sequential(transposeA, transposeB, M, N, K, alpha, A, K, B, N, beta, C, N);

    t2 = MPI_Wtime();

    if (rank == 0){
        printf("Elapsed time is %lf. Computation time is %lf\n", t2-t1, computationTime);
        #ifdef VALIDATE
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N, beta, referenceC, N);
            Dtest_equality(C, referenceC, M*N);
        #endif
    }

    MPI_Finalize();

    return 0;
}
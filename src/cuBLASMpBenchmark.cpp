#include "cuBLASMpWrapped.hpp"

// #define VALIDATE

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    #ifdef DEBUG
        printf("Rank: %d Size: %d\n", rank, size);
    #endif

    /* ./cublasmpTest M N K Mb Nb */
    long int M, N, K, Mb, Nb;
    if (argc < 6) {
        std::cerr << "Usage: ./cublasmpTest M N K Mb Nb" << std::endl;
        exit(1);
    }
    
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
    Mb = atoi(argv[4]);
    Nb = atoi(argv[5]);

    long int lda, ldb, ldc;

    double alpha, beta;
    alpha = 0.213;
    beta = 1.329;

    double *A, *B, *C, *referenceC;

    double generationStart = MPI_Wtime();
    if (rank == 0) {
        A = (double*) malloc(sizeof(double) * M * K);
        B = (double*) malloc(sizeof(double) * N * K);
        C = (double*) malloc(sizeof(double) * M * N);

        generateMatrixGPU(A, M, K);
        generateMatrixGPU(B, N, K);
        generateMatrixGPU(C, M, N);

        #ifdef VALIDATE
            referenceC = copyMatrix(C, M, N);
        #endif
    }
    double generationEnd = MPI_Wtime();

    if (rank == 0)
        printf("Generation Time: %lf\n", generationEnd - generationStart);

    lda = M;
    ldb = K;
    ldc = M;

    int dRow, dCol;

    calculateProcessGrid(&dRow, &dCol, size);

    MPI_Barrier(MPI_COMM_WORLD);

    double t1 = MPI_Wtime();

    cuBLASMpDgemmWrap('n', 'n', M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, Mb, Nb, dRow, dCol);

    double t2 = MPI_Wtime();

    if (rank == 0) {
        printf("Rank: %d Elapsed Time: %lf\n", rank, t2 - t1);

        #ifdef VALIDATE
            for (int i = 0; i < 10; i++)
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, referenceC, ldc);
            Dtest_equality(C, referenceC, M*N);
        #endif
    }

    MPI_Finalize();

    return 0;
};
#include <distributedParalia.hpp>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    long long M, N, K;
    long long lda, ldb, ldc;
    double alpha, beta;
    double *A, *B, *C;

    if (argc < 7) {
        fprintf(stderr, "Usage: ./distributedParaliaValidators '{TransA}{TransB}' M N K alpha beta\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    char transposeA = argv[1][0];
    char transposeB = argv[1][1];
    M = atoll(argv[2]);
    N = atoll(argv[3]);
    K = atoll(argv[4]);
    alpha = atof(argv[5]);
    beta = atof(argv[6]);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    A = (double*) malloc(sizeof(double) * M * K);
    B = (double*) malloc(sizeof(double) * N * K);
    C = (double*) malloc(sizeof(double) * M * N);

    /* Generate Matrices */
    if (rank == 0) {
        MatrixInit(A, M, K, 0);
        MatrixInit(B, K, N, 0);
        MatrixInit(C, M, N, 0);
    }

    /* Not really necessary, but its Column-Major */
    lda = M;
    ldb = K;
    ldc = M;

    validateDistributedSequentialParalia(transposeA, transposeB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return 0;
}

void validateDistributedSequentialParalia(char TransA,  char TransB, const long long M, const long long N, const long long K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double* referenceC;
    if (rank == 0) {
        referenceC = (double*) malloc(sizeof(double) * M * N);
        copyBlock(M, N, C, referenceC, ldC, ldC);
    }
    
    bool logging = false;
    bool warmup = false;
    int numberOfRuns = 1;
    int aLoc = 5, bLoc = 5, cLoc = 5;

    paraliaFullGemmOffloadSequential(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC, numberOfRuns, logging, warmup, aLoc, bLoc, cLoc);
    
    if (rank == 0) {
        /* Run normal PARALiA GEMM */
        PARALiADgemm(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, referenceC, ldC);
        /* Test equality between matrices */
        Dtest_equality(C, referenceC, M*N);
    }

    return;
}
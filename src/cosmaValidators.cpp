#include <cosmaWrapped.hpp>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    long long M, N, K;
    long long lda, ldb, ldc;
    double alpha, beta;
    double *A, *B, *C, *referenceC;
    if (argc < 9) {
        printf("Usage: ./cosmaBenchmark '{TransA}{TransB}' M N K Mb Nb alpha beta\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    char transposeA = argv[1][0];
    char transposeB = argv[1][1];
    M = atoi(argv[2]);
    N = atoi(argv[3]);
    K = atoi(argv[4]);
    int Mb = atoi(argv[5]);
    int Nb = atoi(argv[6]);
    alpha = atof(argv[7]);
    beta = atof(argv[8]);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Generate Matrices */
    if (rank == 0) {
        A = (double*) malloc(sizeof(double) * M * K);
        B = (double*) malloc(sizeof(double) * N * K);
        C = (double*) malloc(sizeof(double) * M * N);
        referenceC = (double*) malloc(sizeof(double) * M * N);
        MatrixInit(A, M, K, 0);
        MatrixInit(B, K, N, 0);
        MatrixInit(C, M, N, 0);
        copyBlock(M, N, C, referenceC, M, M, true);
    }
    /* Column-Major */
    lda = M;
    ldb = K;
    ldc = M;

    bool logging = false;
    bool gatherResults = true;
        
    cosmaFullGemmOffloadScalapack(transposeA, transposeB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, Mb, Nb, MPI_COMM_WORLD, 1, logging, gatherResults);

    if (rank == 0) {
        cblas_dgemm(CblasColMajor, charToCblasTransOp(transposeA), charToCblasTransOp(transposeB), M, N, K, alpha, A, lda, B, ldb, beta, referenceC, ldc);
        Dtest_equality(referenceC, C, M*N);
    }

    /* Run validator */
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
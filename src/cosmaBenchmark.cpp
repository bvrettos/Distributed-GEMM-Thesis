#include <cosmaWrapped.hpp>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    long long M, N, K;
    long long lda, ldb, ldc;
    double alpha, beta;
    double *A, *B, *C;
    bool fullOffload = false;
    if (argc < 8) {
        printf("Usage: ./cosmaBenchmark '{TransA}{TransB}' M N K alpha beta numberOfRuns fullOffload{default=false}\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    
    /* Valid Options */
    if (argc == 9) {
        fullOffload = true;
    }
    else if (argc > 9) {
        printf("Usage: ./cosmaBenchmark '{TransA}{TransB}' M N K alpha beta numberOfRuns fullOffload{default=false}\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    char transposeA = argv[1][0];
    char transposeB = argv[1][1];
    M = atoi(argv[2]);
    N = atoi(argv[3]);
    K = atoi(argv[4]);
    alpha = atof(argv[5]);
    beta = atof(argv[6]);
    int numberOfRuns = atoi(argv[7]);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    C = (double*) malloc(sizeof(double) * M * N);
    lda = M;
    ldb = K;
    ldc = M;

    bool logging = true;
    bool gatherResults = true;

    if (fullOffload) {
        printf("Running Full Offload COSMA Run (%d runs) with M=%lld, N=%lld, K=%lld and %d GPUs\n", numberOfRuns, M, N, K, size);
        A = (double*) malloc(sizeof(double) * M * K);
        B = (double*) malloc(sizeof(double) * N * K);

        /* Generate Matrices */
        if (rank == 0) {
            MatrixInit(A, M, K, 0);
            MatrixInit(B, K, N, 0);
            MatrixInit(C, M, N, 0);
        }
        /* Distribute Matrices ScaLAPACK style */
        // cosmaFullGemmOffloadScalapack(transposeA, transposeB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, Mb, Nb, logging);
    }
    else {
        printf("Running Pre Distributed COSMA Run (%d runs) with M=%lld, N=%lld, K=%lld and %d GPUs\n", numberOfRuns, M, N, K, size);
        cosmaPreDistributedOptimalGemm(transposeA, transposeB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, MPI_COMM_WORLD, numberOfRuns, logging);
        // cosmaPreDistributedScalapackGemm(transposeA, transposeB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, 2048, 2048, MPI_COMM_WORLD, numberOfRuns, logging);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
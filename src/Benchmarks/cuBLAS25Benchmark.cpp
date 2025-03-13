#include <25DCannon.hpp>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    long long M, N, K;
    long long lda, ldb, ldc;
    double alpha, beta;
    double *A, *B, *C, *referenceC;

    if (argc < 12) {
        fprintf(stderr, "Usage: ./cublas25DBenchmark '{TransA}{TransB}' M N K alpha beta dRow dCol C numberOfRuns dataLocation={host,devices} fullOffload{default=false}\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    char transposeA = argv[1][0];
    char transposeB = argv[1][1];
    M = atoi(argv[2]);
    N = atoi(argv[3]);
    K = atoi(argv[4]);
    alpha = atof(argv[5]);
    beta = atof(argv[6]);
    int dRow = atoi(argv[7]);
    int dCol = atoi(argv[8]);
    int c = atoi(argv[9]);
    int numberOfRuns = atoi(argv[10]);
    std::string dataLocation = argv[11];

    bool fullOffload = false;
    if (argc==13) {
        fullOffload = true;
    }
    
    int initialDataLocation;
    if (dataLocation == "host") {
        initialDataLocation = -1;
    } 
    else if (dataLocation == "devices") {
        initialDataLocation = 0;
    }
    else {
        printf("Initial Data location not recognized, check usage\n");
        printf("Usage: ./cublas25DBenchmark '{TransA}{TransB}' M N K alpha beta dRow dCol C numberOfRuns dataLocation={host,devices} fullOffload{default=false}\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    /* Not really necessary, but its Column-Major */
    C = (double*) malloc(sizeof(double) * M * N);
    lda = M;
    ldb = K;
    ldc = M;

    bool logging = true;
    bool gatherResults = false;

    if (fullOffload) {
        printf("Running Full Offload 2.5D SUMMA cuBLAS Run (%d runs) with M=%lld, N=%lld, K=%lld and %d GPUs\n", numberOfRuns, M, N, K, size);
        A = (double*) malloc(sizeof(double) * M * K);
        B = (double*) malloc(sizeof(double) * N * K);

        /* Generate Matrices */
        if (rank == 0) {
            MatrixInit(A, M, K, 0);
            MatrixInit(B, K, N, 0);
            MatrixInit(C, M, N, 0);
        }
        fullOffloadSumma25Dgemm(transposeA, transposeB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, dRow, dCol, c, numberOfRuns, logging, true);
    }
    else {
        printf("Running Pre Distributed 2.5D SUMMA cuBLAS Run (%d runs) with M=%lld, N=%lld, K=%lld and %d GPUs\n", numberOfRuns, M, N, K, size);
        // preDistributedCannon25NCCL(transposeA, transposeB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, dRow, dCol, c, numberOfRuns, logging, gatherResults);
        preDistributedSumma25Dgemm(transposeA, transposeB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, dRow, dCol, c, numberOfRuns, logging, gatherResults);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
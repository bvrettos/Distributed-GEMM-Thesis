#include <cstdio>
#include <cstdlib>
#include <mpi.h>
#include <dplasma.h>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    long long M, N, K;
    long long lda, ldb, ldc;
    double alpha, beta;
    double *A, *B, *C, *referenceC;

    if (argc < 11) {
        fprintf(stderr, "Usage: ./dplasmaBenchmark '{TransA}{TransB}' M N K Mb Nb alpha beta numberOfRuns dataLocation={host,devices} fullOffload{default=false}\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
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
    int numberOfRuns = atoi(argv[9]);
    std::string dataLocation = argv[10];

    bool fullOffload = false;
    if (argc==12) {
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
        printf("Usage: ./slateBenchmark '{TransA}{TransB}' M N K Mb Nb alpha beta numberOfRuns dataLocation={host,devices} fullOffload{default=false}\n");
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
    bool gatherResults = true;
    if (fullOffload) {
        printf("Running Full Offload Slate Run (%d runs) with M=%lld, N=%lld, K=%lld and %d GPUs\n", numberOfRuns, M, N, K, size);
        A = (double*) malloc(sizeof(double) * M * K);
        B = (double*) malloc(sizeof(double) * N * K);

        /* Generate Matrices */
        if (rank == 0) {
            MatrixInit(A, M, K, 0);
            MatrixInit(B, K, N, 0);
            MatrixInit(C, M, N, 0);
        }
        /* Distribute Matrices ScaLAPACK style */
        slateFullGemmOffload(transposeA, transposeB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, Mb, Nb, logging, gatherResults, initialDataLocation);
    }
    else {
        printf("Running Pre Distributed Slate Run (%d runs) with M=%lld, N=%lld, K=%lld and %d GPUs\n", numberOfRuns, M, N, K, size);
        slatePreDistributedGemm(transposeA, transposeB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, Mb, Nb, numberOfRuns, logging, gatherResults, initialDataLocation);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
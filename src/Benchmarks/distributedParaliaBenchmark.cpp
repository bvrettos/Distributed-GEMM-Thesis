#include <distributedParalia.hpp>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    long long M, N, K;
    double alpha, beta;
    int numberOfRuns;
    char TransposeA, TransposeB;
    int aLoc, bLoc, cLoc;

    if (argc < 11) {
        std::cerr << "Usage ./distributedParaliaBenchmark {TransA}{TransB} M N K alpha beta A_loc B_loc C_loc numberOfRuns" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    TransposeA = argv[1][0];
    TransposeB = argv[1][1];
    M = atoll(argv[2]);
    N = atoll(argv[3]);
    K = atoll(argv[4]);
    alpha = atof(argv[5]);
    beta = atof(argv[6]);
    aLoc = atoi(argv[7]);
    bLoc = atoi(argv[8]);
    cLoc = atoi(argv[9]);
    numberOfRuns = atoi(argv[10]);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long lda, ldb, ldc;
    lda = M;
    ldb = K;
    ldc = M;

    bool logging = true;
    bool gatherResults = true;
    double *A, *B, *C;

    /* For results to be gathered in or for full offload */
    C = (double*) malloc(sizeof(double) * M * N);

    preDistributedSequentialParaliaGemm(TransposeA, TransposeB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc,
     numberOfRuns, logging, gatherResults, aLoc, bLoc, cLoc);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
#include "cuBLASMpWrapped.hpp"

// #define VALIDATE
// #define DEBUG

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long M, N, K, Mb, Nb;
    double alpha, beta;
    int numberOfRuns;

    if (argc < 10) {
        std::cerr << "Usage: ./cublasmpBenchmark {TransA}{TransB} M N K Mb Nb alpha beta numberOfRuns fullOffload{false if empty}" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    
    char TransposeA = (argv[1][0]);
    char TransposeB = (argv[1][1]);
    M = atoi(argv[2]);
    N = atoi(argv[3]);
    K = atoi(argv[4]);
    Mb = atoi(argv[5]);
    Nb = atoi(argv[6]);
    alpha = atof(argv[7]);
    beta = atof(argv[8]);
    numberOfRuns = atoi(argv[9]);
    bool fullOffload = false;
    if (argc==11) 
        fullOffload = true;

    long long lda, ldb, ldc;
    double *A, *B, *C;

    bool logging = true;
    bool gatherResults = false;

    lda = M;
    ldb = K;
    ldc = M;

    if (fullOffload) {
        if (rank == 0) {
            A = (double*) malloc(sizeof(double) * M * K);
            B = (double*) malloc(sizeof(double) * N * K);
            C = (double*) malloc(sizeof(double) * M * N);
            MatrixInit(A, M, K, 0);
            MatrixInit(B, K, N, 0);
            MatrixInit(C, M, N, 0);    
        }
        cuBLASMpFullGemmOffload(TransposeA, TransposeB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, Mb, Nb, numberOfRuns, logging, gatherResults);    
    }
    else {
        if (rank == 0)
            C = (double*) malloc(sizeof(double) * M * N);
        cuBLASMpPreDistributedGemm(TransposeA, TransposeB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, Mb, Nb, numberOfRuns, logging, gatherResults);
    }
    
    MPI_Finalize();

    return 0;
};
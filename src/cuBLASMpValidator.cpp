#include <cuBLASMpWrapped.hpp>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank,size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long M, N, K, Mb, Nb;
    double alpha, beta;

    if (argc < 9) {
        fprintf(stderr,"Usage: ./cublasmpValidator {TransA}|{TransB} M N K Mb Nb alpha beta\n");
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

    long long lda, ldb, ldc;
    double *A, *B, *C, *referenceC;

    if (rank == 0) {
        A = (double*) malloc(sizeof(double) * M * K);
        B = (double*) malloc(sizeof(double) * N * K);
        C = (double*) malloc(sizeof(double) * M * N);
        MatrixInit(A, M, K, 0);
        MatrixInit(B, K, N, 0);
        MatrixInit(C, M, N, 0);    
        referenceC = copyMatrix(C, M, N);
    }

    lda = M;
    ldb = K;
    ldc = M;

    cuBLASMpGEMMWrap(TransposeA, TransposeB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, Mb, Nb);

    if (rank == 0) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, referenceC, ldc);
        Dtest_equality(C, referenceC, M*N);
    }
    
    MPI_Finalize();

    return 0;
};
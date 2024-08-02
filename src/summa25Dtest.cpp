#include <25DSumma.hpp>


int main(int argc, char* argv[])
{   
    MPI_Init(&argc, &argv);

    if (argc < 9) {
        printf("Usage: srun summa25D M N K alpha beta dRow dCol c");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    long long M = atoll(argv[1]);
    long long N = atoll(argv[2]);
    long long K = atoll(argv[3]);
    double alpha = atof(argv[4]);
    double beta = atof(argv[5]);
    int dRow = atoi(argv[6]);
    int dCol = atoi(argv[7]);
    int c = atoi(argv[8]);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Initialize A,B,C matrices */
    double *A, *B, *C;
    double* refC;

    long long lda = M;
    long long ldb = K;
    long long ldc = M;
    
    if (rank == 0) {
        A = (double*) malloc(sizeof(double) * M * K);
        B = (double*) malloc(sizeof(double) * N * K);
        C = (double*) malloc(sizeof(double) * M * N);
        refC = (double*) malloc(sizeof(double) * M * N);
        MatrixInit(A, M, K, 0);
        MatrixInit(B, K, N, 0);
        MatrixInit(C, M, N, 0);    
        copyBlock(M, N, C, refC, M, M, true);
        printf("Testing 2.5D SUMMA with M=N=K=%lld and grid of %dx%dx%d\n", M, dRow, dCol, c);
    }

    Summa25Decomposer decomposer(M, N, K, dRow, dCol, c);

    /* Every important call happens on multiply */
    double before, after;
    before = MPI_Wtime();
    decomposer.multiply('N', 'N', A, B, C, alpha, beta);
    after = MPI_Wtime();

    if (rank == 0) {
        printf("Multiply time: %lf\n", after-before);
    }

    decomposer.reorderMatrix(0, C);
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        cublasHandle_t cublasContext;
        double* devA, *devB, *devC;
        cudaSetDevice(0);
        cudaMalloc((void**)&devA, sizeof(double) * M * K);
        cudaMalloc((void**)&devB, sizeof(double) * N * K);
        cudaMalloc((void**)&devC, sizeof(double) * M * N);
        cudaMemcpy(devA, A, sizeof(double) * M * K, cudaMemcpyHostToDevice);
        cudaMemcpy(devB, B, sizeof(double) * N * K, cudaMemcpyHostToDevice);
        cudaMemcpy(devC, refC, sizeof(double) * M * N, cudaMemcpyHostToDevice);
        CUBLAS_CHECK(cublasCreate(&cublasContext));
        CUBLAS_CHECK(cublasDgemm(cublasContext, charToCublasTransOp('N'), charToCublasTransOp('N'), M, N, K, &alpha, devA, lda, devB, ldb, &beta, devC, ldc));
        CUBLAS_CHECK(cublasDestroy(cublasContext));

        cudaMemcpy(refC, devC, sizeof(double) * M * N, cudaMemcpyDeviceToHost);
        Dtest_equality(refC, C, M*N);
    }

    MPI_Finalize();
    return 0;
}
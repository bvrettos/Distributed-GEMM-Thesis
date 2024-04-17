#include "mpiBLAS_wrappers.hpp"
/*  
    Title: Optimizing Matrix-Matrix Multiplication for Multi-Node Multi-GPU (Supercomputer) environments

    NECESSARY: IMPLEMENT CACHING FOR MORE STRAIGHT FORWARD BENCHMARKING
    0. 2D Block Sequential Decomposition DONE
    1. 2D Block Cyclic Decomposition  
    2. BLAS to PBLAS translator 
    3. cuBLASMp Wrapper (to compare against) DONE (Bonus: make distribution more CUDA aware)
    4. CUDA Aware MPI Optimizations 
    
    Bonus:
        5. PBLAS to BLAS translator (8a doume)
*/

/*
    Peiramata:
        1. cuBLASMp {GPUs Per node: 1, 2, 4}{NodeNumbers: 1, 2, 4, 8}{M=N=K: 4096 - 2^16: Step 4096}
        2. 2D Block Sequential idia parameters

        Warmup 10 runs - then run 10 runs
*/

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int M, N, K;
    int lda, ldb, ldc;
    double alpha, beta;
    double *A, *B, *C, *referenceC;

    if (argc < 7) {
        printf("Usage: ./paraliaBenchmark 'NN' M N K alpha beta");
        exit(1);
    }
    char transposeA = argv[1][0];
    char transposeB = argv[1][1];
    M = atoi(argv[2]);
    N = atoi(argv[3]);
    K = atoi(argv[4]);
    
    alpha = atof(argv[5]);
    beta = atof(argv[6]);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        A = (double*) malloc(sizeof(double) * M * K);
        B = (double*) malloc(sizeof(double) * N * K);
        C = (double*) malloc(sizeof(double) * M * N);
        MatrixInit(A, M, K, 0);
        MatrixInit(B, K, N, 0);
        MatrixInit(C, M, N, 0);    
        #ifdef VALIDATE
            referenceC = copyMatrix(C, M, N);
        #endif
    }
    
    lda = M;
    ldb = K;
    ldc = M;

    double elapsedTime = PARALiA_MPI_Dgemm('n', 'n', M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    if (rank == 0) {
        #ifdef VALIDATE
            for (int i = 0; i < 1; i++)
                cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, referenceC, ldc);
            Dtest_equality(C, referenceC, M*N);
        #endif
    }

    MPI_Finalize();
    
    return 0;
}
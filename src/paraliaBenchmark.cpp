#include "mpiBLAS_wrappers.hpp"
/*  
    Title: Optimizing Matrix-Matrix Multiplication for Multi-Node Multi-GPU (Supercomputer) environments

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
    double *A, *B, *C;

    if (argc < 4) {
        printf("Usage: ./paraliaBenchmark M N K");
        exit(1);
    }

    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);

    int rank, size;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        A = (double*) malloc(sizeof(double)*M*K);
        B = (double*) malloc(sizeof(double)*N*K);
        C = (double*) malloc(sizeof(double)*M*N);

        generateMatrixGPU(A, M, K);
        generateMatrixGPU(B, K, N);
        generateMatrixGPU(C, M, N);
    }

    lda = M;
    ldb = K;
    ldc = M;

    double elapsedTime = PARALiA_MPI_Dgemm('n', 'n', M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    MPI_Finalize();
    
    return 0;
}
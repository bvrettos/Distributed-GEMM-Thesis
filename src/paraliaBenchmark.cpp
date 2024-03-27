#include "mpiBLAS_wrappers.hpp"
#include <PARALiA.hpp>

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
    int m, n, k;
    int lda, ldb, ldc;
    double alpha, beta;
    double *A, *B, *C;
    int dev_ids[] = {1, 2};

    printf("Hello\n");
    double elapsedTime = PARALiA_MPI_Dgemm('n', 'n', m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, dev_ids);
    
    return 0;
}
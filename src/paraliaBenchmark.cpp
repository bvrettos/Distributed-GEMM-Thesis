#include "mpiBLAS_wrappers.hpp"
#include <PARALiA.hpp>

/*  
    Title: Optimizing Matrix-Matrix Multiplication for Multi-Node Multi-GPU (Supercomputer) environments

    0. 2D Block Sequential Decomposition
    1. 2D Block Cyclic Decomposition 
    2. BLAS to PBLAS translator
    3. cuBLASMp Wrapper (to compare against)
    4. CUDA Aware MPI Optimizations

    Bonus:
        5. PBLAS to BLAS translator (8a doume)
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
#include "cuBLASMP_wrappers.hpp"

#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <vector>
#include <cublasmp.h>
#include <validation.hpp>
#include <cblas.h>
#include <unistd.h>

// #define VALIDATE

/* Validation: Check testing.cpp validation */

    /* CUBLASMP STEPS:
        1. Find rank and size of world - DONE
        2. Find ID of localGPU - DONE
        3. Create CAL communicator - DONE
        4. Create stream for localGPU - DONE
        5. Create handle for CUBLASMP - DONE
        6. Initialize pointers for both matrices: 
            6a. global DONE
            6b. local DONe
            6c. device DONe
        7. Call cublasMpNumroc to find size to allocate on Local matrix DONE
        8. Copy data from host to device DONE
        9. Create grid with cublasMpGridCreate DONE
        10. Create Matrix descriptors DONE
        11. Allocate necessary memory using... DONE
        12. Sync processes DONE
        13. call cublasMpGemm DONE
        14. Destroy everything DONE
*/

void cuBLASMpDgemmWrap(char TransA,  char TransB, long int M, long int N, long int K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC, int Mb, int Nb, int dRow, int dCol)
{
    char orientation = 'r';
    int ia = 1, ja = 1, ib = 1, jb = 1, ic = 1, jc = 1;

    /* 1. Find rank and size of World */
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* 2. Find ID of localGPU */
    int localDeviceID = rank % (dRow*dCol);
    CUDA_CHECK(cudaSetDevice(localDeviceID));

    int localRow = (orientation == 'c' ? rank % dRow : rank / dCol);
    int localCol = (orientation == 'c' ? rank / dRow : rank % dCol);

    /* 3. Create CAL Communicator */
    cal_comm_t calCommunicator = createCalCommunicator(rank, size, localDeviceID);

    /* 4. Create Stream for Local GPU */
    cudaStream_t stream = NULL;
    CUDA_CHECK(cudaStreamCreate(&stream));
    
    /* 5. Create handle for cuBLASMp */
    cublasMpHandle_t handle = NULL;
    CUBLAS_CHECK(cublasMpCreate(&handle, stream));

    /* 6. Initialize Pointers and Matrix Handlers */
    cublasMpMatrixDescriptor_t descA, descB, descC;
    double *d_A, *d_B, *d_C, *d_work;

    size_t workspaceInBytesOnDevice = 0;
    size_t workspaceInBytesOnHost = 0;

    /* Decompose Matrices */    
    pblasDecomposer decomposerA(M, K, Mb, Nb, dRow, dCol, A, MPI_COMM_WORLD);
    pblasDecomposer decomposerB(K, N, Mb, Nb, dRow, dCol, B, MPI_COMM_WORLD);
    pblasDecomposer decomposerC(M, N, Mb, Nb, dRow, dCol, C, MPI_COMM_WORLD);

    const int64_t llda = decomposerA.localRows;
    const int64_t loc_n_a = decomposerA.localColumns;

    const int64_t lldb = decomposerB.localRows;
    const int64_t loc_n_b = decomposerB.localColumns;

    const int64_t lldc = decomposerC.localRows;
    const int64_t loc_n_c = decomposerC.localColumns;

    double *localA, *localB, *localC;
    localA = decomposerA.localMatrix;
    localB = decomposerB.localMatrix;
    localC = decomposerC.localMatrix;

    CUDA_CHECK(cudaMallocAsync(&d_A, llda * loc_n_a *sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_B, lldb * loc_n_b *sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_C, lldc * loc_n_c *sizeof(double), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A, localA, llda * loc_n_a * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, localB, lldb * loc_n_b * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C, localC, lldc * loc_n_c * sizeof(double), cudaMemcpyHostToDevice, stream));

    /* 9. Create Grid */
    cublasMpGrid_t grid = NULL;
    CUBLAS_CHECK(cublasMpGridCreate(
        handle,
        dRow,
        dCol,
        CUBLASMP_GRID_LAYOUT_ROW_MAJOR,
        calCommunicator,
        &grid));

    /* 10. Create Matrix Descriptors */
    CUBLAS_CHECK(
        cublasMpMatrixDescriptorCreate(handle, M, K, Mb, Nb, 0, 0, llda, CUDA_R_64F, grid, &descA));
    CUBLAS_CHECK(
        cublasMpMatrixDescriptorCreate(handle, K, N, Mb, Nb, 0, 0, lldb, CUDA_R_64F, grid, &descB));
    CUBLAS_CHECK(
        cublasMpMatrixDescriptorCreate(handle, M, N, Mb, Nb, 0, 0, lldc, CUDA_R_64F, grid, &descC));

    /* 11. Calculate necessary memory size */
    CUBLAS_CHECK(cublasMpGemm_bufferSize(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        M,
        N,
        K,
        &alpha,
        d_A,
        ia,
        ja,
        descA,
        d_B,
        ib,
        jb,
        descB,
        &beta,
        d_C,
        ic,
        jc,
        descC,
        CUDA_R_64F,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost));

    CUDA_CHECK(cudaMallocAsync(&d_work, workspaceInBytesOnDevice, stream));
    std::vector<int8_t> h_work(workspaceInBytesOnHost);

    CAL_CHECK(cal_stream_sync(calCommunicator, stream));
    CAL_CHECK(cal_comm_barrier(calCommunicator, stream));

    const double begin = MPI_Wtime();
    CUBLAS_CHECK(cublasMpGemm(
        handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        M,
        N,
        K,
        &alpha,
        d_A,
        ia,
        ja,
        descA,
        d_B,
        ib,
        jb,
        descB,
        &beta,
        d_C,
        ic,
        jc,
        descC,
        CUDA_R_64F,
        d_work,
        workspaceInBytesOnDevice,
        h_work.data(),
        workspaceInBytesOnHost));

    CUDA_CHECK(cudaMemcpyAsync(localC, d_C, lldc * loc_n_c * sizeof(double), cudaMemcpyDeviceToHost, stream));

    CAL_CHECK(cal_stream_sync(calCommunicator, stream));
    CAL_CHECK(cal_comm_barrier(calCommunicator, stream));

    decomposerC.gatherMatrix();

    const double end = MPI_Wtime();

    if (rank == 0)
        printf("Rank: %d Duration: %lf GFlops: %lf\n", rank,  end - begin, (2 * M * N * K * 1e-9) / (end - begin));

    /* 14. Destroy Everything */
    CUBLAS_CHECK(cublasMpMatrixDescriptorDestroy(handle, descA));
    CUBLAS_CHECK(cublasMpMatrixDescriptorDestroy(handle, descB));
    CUBLAS_CHECK(cublasMpMatrixDescriptorDestroy(handle, descC));

    CUBLAS_CHECK(cublasMpGridDestroy(handle, grid));

    CUBLAS_CHECK(cublasMpDestroy(handle));

    CUDA_CHECK(cudaFreeAsync(d_A, stream));
    CUDA_CHECK(cudaFreeAsync(d_B, stream));
    CUDA_CHECK(cudaFreeAsync(d_C, stream));
    CUDA_CHECK(cudaFreeAsync(d_work, stream));

    CAL_CHECK(cal_comm_barrier(calCommunicator, stream));

    CAL_CHECK(cal_comm_destroy(calCommunicator));

    CUDA_CHECK(cudaStreamDestroy(stream));

    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int M, N, K;
    int lda, ldb, ldc;
    M = 8192*2;
    N = 8192*2;
    K = 8192*2;

    double alpha, beta;
    alpha = 0.213;
    beta = 1.329;

    double *A, *B, *C, *referenceC;

    if (rank == 0) {
        A = (double*) malloc(sizeof(double) * M * K);
        B = (double*) malloc(sizeof(double) * N * K);
        C = (double*) malloc(sizeof(double) * M * N);

        generateMatrixColumnMajor(A, M, K);
        generateMatrixColumnMajor(B, N, K);
        generateMatrixColumnMajor(C, M, N);

        referenceC = copyMatrix(C, M, N);
    }

    lda = M;
    ldb = K;
    ldc = M;

    int Mb, Nb;
    int dRow, dCol;

    Mb = 512;
    Nb = 512;

    dRow = 2;
    dCol = 1;

    MPI_Barrier(MPI_COMM_WORLD);

    double t1 = MPI_Wtime();

    cuBLASMpDgemmWrap('c', 'c', M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, Mb, Nb, dRow, dCol);

    double t2 = MPI_Wtime();

    printf("Rank: %d Elapsed Time: %lf\n", rank, t2 - t1);

    if (rank == 0) {
        #ifdef VALIDATE
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, referenceC, ldc);
            Dtest_equality(C, referenceC, M*N);
        #endif
    }

    MPI_Finalize();

    return 0;
};
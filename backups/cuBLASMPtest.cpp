#include "cuBLASMP_wrappers.hpp"
#include "cmatrix.h"

#include <assert.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <vector>
#include <cublasmp.h>

/* Validation: Check testing.cpp validation */

    /* CUBLASMP STEPS:
        1. Find rank and size of world - DONE
        2. Find ID of localGPU - DONE
        3. Create CAL communicator - DONE
        4. Create stream for localGPU - DONE
        5. Create handle for CUBLASMP - DONE
        6. Initialize pointers for both matrices:
            6a. global
            6b. local
            6c. device
        7. Call cublasMpNumroc to ...
        8. Copy data from host to device
        9. Create grid with cublasMpGridCreate DONE
        10. Create Matrix descriptors DONE
        11. Allocate necessary memory using... DONE
        12. Sync processes
        13. call cublasMpGemm
        14. Destroy everything DONE
    */

// void cuBLASMpDgemmWrap(char TransA,  char TransB, long int M, long int N, long int K,
//   double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
//   long int ldC, short numberOfDevices, int dev_ids[])
// {
//     char orientation = 'c';

//     /* 1. Find rank and size of World */
//     int rank, size;
//     MPI_Comm_size(MPI_COMM_WORLD, &size);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
//     /* BONUS STEP: calculate dRow, dCol */
//     int dRow, dCol;
//     calculateGridDimensions(numberOfDevices, dRow, dCol);

//     /* 2. Find ID of localGPU */
//     int localDeviceID = getLocalDevice();
//     CUDA_CHECK(cudaSetDevice(local_device));

//     int localRow = (orientation == 'c' ? rank % nprow : rank / npcol);
//     int localCol = (orientation == 'c' ? rank / nprow : rank % npcol);

//     /* 3. Create CAL Communicator */
//     cal_comm_t calCommunicator = createCalCommunicator(rank, size, localDeviceID);

//     /* 4. Create Stream for Local GPU */
//     cudaStream_t stream = NULL;
//     CUDA_CHECK(cudaStreamCreate(&stream));
    
//     /* 5. Create handle for cuBLASMp */
//     cublasMpHandle_t handle = NULL;
//     CUBLAS_CHECK(cublasMpCreate(&handle, stream));

//     /* 6. Initialize Pointers and Matrix Handlers */
//     cublasMpMatrixDescriptor_t descA, descB, descC;
//     double *d_A, *d_B, *d_C, *d_work;

//     size_t workspaceInBytesOnDevice = 0;
//     size_t workspaceInBytesOnHost = 0;

//     /* 9. Create Grid */
//     cublasMpGrid_t grid = NULL;
//     CUBLAS_CHECK(cublasMpGridCreate(
//         handle,
//         dRow,
//         dCol,
//         orientation == 'c' ? CUBLASMP_GRID_LAYOUT_COL_MAJOR : CUBLASMP_GRID_LAYOUT_ROW_MAJOR,
//         cal_comm,
//         &grid));

//     /* 10. Create Matrix Descriptors */
//     CUBLAS_CHECK(
//         cublasMpMatrixDescriptorCreate(handle, global_m_a, global_n_a, mbA, nbA, 0, 0, llda, CUDA_R_64F, grid, &descA));
//     CUBLAS_CHECK(
//         cublasMpMatrixDescriptorCreate(handle, global_m_b, global_n_b, mbB, nbB, 0, 0, lldb, CUDA_R_64F, grid, &descB));
//     CUBLAS_CHECK(
//         cublasMpMatrixDescriptorCreate(handle, global_m_c, global_n_c, mbC, nbC, 0, 0, lldc, CUDA_R_64F, grid, &descC));

//     /* 11. Calculate necessary memory size */
//     CUBLAS_CHECK(cublasMpGemm_bufferSize(
//         handle,
//         CUBLAS_OP_N,
//         CUBLAS_OP_N,
//         m,
//         n,
//         k,
//         &alpha,
//         d_A,
//         ia,
//         ja,
//         descA,
//         d_B,
//         ib,
//         jb,
//         descB,
//         &beta,
//         d_C,
//         ic,
//         jc,
//         descC,
//         CUDA_R_64F,
//         &workspaceInBytesOnDevice,
//         &workspaceInBytesOnHost));

//     /* 14. Destroy Everything */
//     CUBLAS_CHECK(cublasMpMatrixDescriptorDestroy(handle, descA));
//     CUBLAS_CHECK(cublasMpMatrixDescriptorDestroy(handle, descB));
//     CUBLAS_CHECK(cublasMpMatrixDescriptorDestroy(handle, descC));

//     CUBLAS_CHECK(cublasMpGridDestroy(handle, grid));

//     CUBLAS_CHECK(cublasMpDestroy(handle));

//     CUDA_CHECK(cudaFreeAsync(d_A, stream));
//     CUDA_CHECK(cudaFreeAsync(d_B, stream));
//     CUDA_CHECK(cudaFreeAsync(d_C, stream));
//     CUDA_CHECK(cudaFreeAsync(d_work, stream));

//     CAL_CHECK(cal_comm_barrier(calCommunicator, stream));

//     CAL_CHECK(cal_comm_destroy(calCommunicator));

//     CUDA_CHECK(cudaStreamDestroy(stream));

//     MPI_Barrier(MPI_COMM_WORLD);
// }

int main(int argc, char* argv[])
{

    // MPI_Init(NULL, NULL);

    // const int64_t m = 10;
    // const int64_t n = 10;
    // const int64_t k = 10;
    // const int64_t ia = 3;
    // const int64_t ja = 3;
    // const int64_t ib = 3;
    // const int64_t jb = 1;
    // const int64_t ic = 1;
    // const int64_t jc = 1;
    // const int64_t mbA = 2;
    // const int64_t nbA = 2;
    // const int64_t mbB = 2;
    // const int64_t nbB = 2;
    // const int64_t mbC = 2;
    // const int64_t nbC = 2;

    // const int nprow = 2; //dRow
    // const int npcol = 1; //dColumn
    
    // int rank, nranks;

    // MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // const int myprow = ('c' == 'c' ? rank % nprow : rank / npcol);
    // const int mypcol = ('c' == 'c' ? rank / nprow : rank % npcol);

    // int localRank, local_device, deviceCount;
    // getLocalDevice(&localRank, &deviceCount, &local_device);
    // CUDA_CHECK(cudaSetDevice(local_device));
    // CUDA_CHECK(cudaFree(nullptr));

    // cal_comm_t cal_comm = createCalCommunicator(rank, nranks, local_device);

    // cudaStream_t stream = nullptr;
    // CUDA_CHECK(cudaStreamCreate(&stream));

    // cublasMpHandle_t handle = nullptr;
    // CUBLAS_CHECK(cublasMpCreate(&handle, stream));

    // cublasMpGrid_t grid = nullptr;

    // cublasMpMatrixDescriptor_t descA = nullptr;
    // cublasMpMatrixDescriptor_t descB = nullptr;
    // cublasMpMatrixDescriptor_t descC = nullptr;

    // double* d_A = nullptr;
    // double* d_B = nullptr;
    // double* d_C = nullptr;

    // double* d_work = nullptr;

    // double alpha = 1.0;
    // double beta = 1.0;

    // size_t workspaceInBytesOnDevice = 0;
    // size_t workspaceInBytesOnHost = 0;

    // const int64_t global_m_a = (ia - 1) + m;
    // const int64_t global_n_a = (ja - 1) + k;
    // const int64_t global_m_b = (ib - 1) + k;
    // const int64_t global_n_b = (jb - 1) + n;
    // const int64_t global_m_c = (ic - 1) + m;
    // const int64_t global_n_c = (jc - 1) + n;

    int64_t test = cublasMpNumroc(8, 3, 0, 0, 2);
    int64_t test2 = cublasMpNumroc(8, 2, 0, 0, 2);
    std::cout << test << " "<< test2 << std::endl;

    // const int64_t llda = cublasMpNumroc(global_m_a, mbA, myprow, 0, nprow);
    // const int64_t loc_n_a = cublasMpNumroc(global_n_a, nbA, mypcol, 0, npcol);

    // const int64_t lldb = cublasMpNumroc(global_m_b, mbB, myprow, 0, nprow);
    // const int64_t loc_n_b = cublasMpNumroc(global_n_b, nbB, mypcol, 0, npcol);

    // const int64_t lldc = cublasMpNumroc(global_m_c, mbC, myprow, 0, nprow);
    // const int64_t loc_n_c = cublasMpNumroc(global_n_c, nbC, mypcol, 0, npcol);

    // std::vector<double> h_A(llda * loc_n_a, 0);
    // std::vector<double> h_B(lldb * loc_n_b, 0);
    // std::vector<double> h_C(lldc * loc_n_c, 0);

    // std::cout << h_A.size() << " " << h_B.size() << " " << h_C.size() << std::endl;

    // generate_random_matrix(m, k, h_A.data(), mbA, nbA, ia, ja, llda, nprow, npcol, myprow, mypcol);
    // generate_random_matrix(k, n, h_B.data(), mbB, nbB, ib, jb, lldb, nprow, npcol, myprow, mypcol);
    // generate_random_matrix(m, n, h_C.data(), mbC, nbC, ic, jc, lldc, nprow, npcol, myprow, mypcol);

    // CUDA_CHECK(cudaMallocAsync(&d_A, llda * loc_n_a * sizeof(double), stream));
    // CUDA_CHECK(cudaMallocAsync(&d_B, lldb * loc_n_b * sizeof(double), stream));
    // CUDA_CHECK(cudaMallocAsync(&d_C, lldc * loc_n_c * sizeof(double), stream));

    // CUDA_CHECK(cudaMemcpyAsync(d_A, h_A.data(), llda * loc_n_a * sizeof(double), cudaMemcpyHostToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_B, h_B.data(), lldb * loc_n_b * sizeof(double), cudaMemcpyHostToDevice, stream));
    // CUDA_CHECK(cudaMemcpyAsync(d_C, h_C.data(), lldc * loc_n_c * sizeof(double), cudaMemcpyHostToDevice, stream));

    // CUBLAS_CHECK(cublasMpGridCreate(
    //     handle,
    //     nprow,
    //     npcol,
    //     'c' == 'c' ? CUBLASMP_GRID_LAYOUT_COL_MAJOR : CUBLASMP_GRID_LAYOUT_ROW_MAJOR,
    //     cal_comm,
    //     &grid));

    // CUBLAS_CHECK(
    //     cublasMpMatrixDescriptorCreate(handle, global_m_a, global_n_a, mbA, nbA, 0, 0, llda, CUDA_R_64F, grid, &descA));
    // CUBLAS_CHECK(
    //     cublasMpMatrixDescriptorCreate(handle, global_m_b, global_n_b, mbB, nbB, 0, 0, lldb, CUDA_R_64F, grid, &descB));
    // CUBLAS_CHECK(
    //     cublasMpMatrixDescriptorCreate(handle, global_m_c, global_n_c, mbC, nbC, 0, 0, lldc, CUDA_R_64F, grid, &descC));

    // CUBLAS_CHECK(cublasMpGemm_bufferSize(
    //     handle,
    //     CUBLAS_OP_N,
    //     CUBLAS_OP_N,
    //     m,
    //     n,
    //     k,
    //     &alpha,
    //     d_A,
    //     ia,
    //     ja,
    //     descA,
    //     d_B,
    //     ib,
    //     jb,
    //     descB,
    //     &beta,
    //     d_C,
    //     ic,
    //     jc,
    //     descC,
    //     CUDA_R_64F,
    //     &workspaceInBytesOnDevice,
    //     &workspaceInBytesOnHost));

    // CUDA_CHECK(cudaMallocAsync(&d_work, workspaceInBytesOnDevice, stream));

    // std::vector<int8_t> h_work(workspaceInBytesOnHost);

    // CAL_CHECK(cal_stream_sync(cal_comm, stream));
    // CAL_CHECK(cal_comm_barrier(cal_comm, stream));

    // const double begin = MPI_Wtime();

    // CUBLAS_CHECK(cublasMpGemm(
    //     handle,
    //     CUBLAS_OP_N,
    //     CUBLAS_OP_N,
    //     m,
    //     n,
    //     k,
    //     &alpha,
    //     d_A,
    //     ia,
    //     ja,
    //     descA,
    //     d_B,
    //     ib,
    //     jb,
    //     descB,
    //     &beta,
    //     d_C,
    //     ic,
    //     jc,
    //     descC,
    //     CUDA_R_64F,
    //     d_work,
    //     workspaceInBytesOnDevice,
    //     h_work.data(),
    //     workspaceInBytesOnHost));

    // CAL_CHECK(cal_stream_sync(cal_comm, stream));
    // CAL_CHECK(cal_comm_barrier(cal_comm, stream));

    // const double end = MPI_Wtime();

    // printf("Duration: %lf GFlops: %lf\n", end - begin, (2 * m * n * k * 1e-9) / (end - begin));

    // CUBLAS_CHECK(cublasMpMatrixDescriptorDestroy(handle, descA));
    // CUBLAS_CHECK(cublasMpMatrixDescriptorDestroy(handle, descB));
    // CUBLAS_CHECK(cublasMpMatrixDescriptorDestroy(handle, descC));

    // CUBLAS_CHECK(cublasMpGridDestroy(handle, grid));

    // CUBLAS_CHECK(cublasMpDestroy(handle));

    // CUDA_CHECK(cudaFreeAsync(d_A, stream));
    // CUDA_CHECK(cudaFreeAsync(d_B, stream));
    // CUDA_CHECK(cudaFreeAsync(d_C, stream));
    // CUDA_CHECK(cudaFreeAsync(d_work, stream));

    // CAL_CHECK(cal_comm_barrier(cal_comm, stream));

    // CAL_CHECK(cal_comm_destroy(cal_comm));

    // CUDA_CHECK(cudaStreamDestroy(stream));

    // MPI_Barrier(MPI_COMM_WORLD);

    // MPI_Finalize();

    return 0;
};
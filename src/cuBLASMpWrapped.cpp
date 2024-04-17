#include "cuBLASMpWrapped.hpp"

// #define DEBUG
void cuBLASMpDgemmWrap(char TransA,  char TransB, long int M, long int N, long int K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC, long int Mb, long int Nb, int dRow, int dCol)
{
    int ia = 1, ja = 1, ib = 1, jb = 1, ic = 1, jc = 1;

    /* 1. Find rank and size of World */
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Setup logfile */
    FILE* logfile;
    if (rank == 0) {
        std::string machineName = MACHINE_NAME;
        std::string filename = "DGEMM_execution_logs-" + machineName + "-cuBLASMp.csv";
        std::string header = "Algo,M,N,K,TileRows,TileColumns,dRow,dCol,TotalNodes,TotalGPUs,DecompositionTime,ExecutionTime,GFlops";
        logfile = createLogCsv(filename, header);
    }

    #ifdef DEBUG
        int deviceSize = 0;
        CUDA_CHECK(cudaGetDeviceCount(&deviceSize));
        printf("Rank: %d, Device Count: %d, World Size: %d", rank, deviceSize, size);
    #endif

    /* 2. Find ID of localGPU - On Meluxina, each task gets 1 gpu, so each deviceID is 0 */
    int localDeviceID = rank % 2;
    CUDA_CHECK(cudaSetDevice(localDeviceID));

    int localRow = rank / dCol;
    int localCol = rank % dCol;

    /* 3. Create CAL Communicator */
    double communicatorStart = MPI_Wtime();
    cal_comm_t calCommunicator = createCalCommunicator(rank, size, localDeviceID);
    double communicatorEnd = MPI_Wtime();

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

    double decompositionStart = MPI_Wtime();

    /* Decompose Matrices */    
    pblasDecomposer decomposerA(M, K, Mb, Nb, dRow, dCol, A, MPI_COMM_WORLD);
    pblasDecomposer decomposerB(K, N, Mb, Nb, dRow, dCol, B, MPI_COMM_WORLD);
    pblasDecomposer decomposerC(M, N, Mb, Nb, dRow, dCol, C, MPI_COMM_WORLD);

    double decompositionEnd = MPI_Wtime();

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

    double memcpyStart = MPI_Wtime();

    CUDA_CHECK(cudaMallocAsync(&d_A, llda * loc_n_a *sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_B, lldb * loc_n_b *sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_C, lldc * loc_n_c *sizeof(double), stream));

    CUDA_CHECK(cudaMemcpyAsync(d_A, localA, llda * loc_n_a * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, localB, lldb * loc_n_b * sizeof(double), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_C, localC, lldc * loc_n_c * sizeof(double), cudaMemcpyHostToDevice, stream));

    double memcpyEnd = MPI_Wtime();

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

    for (int i = 0; i < 1; i++) {
        double executionStart = MPI_Wtime();
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
        double executionEnd = MPI_Wtime();

        CAL_CHECK(cal_stream_sync(calCommunicator, stream));
        CAL_CHECK(cal_comm_barrier(calCommunicator, stream));

        if (rank == 0) {
           // printf("Communicator Time: %lf, Decomposition Time: %lf Memcpy Time: %lf\n", communicatorEnd-communicatorStart, decompositionEnd-decompositionStart, memcpyEnd-memcpyStart);
            double executionTime = executionEnd - executionStart;
            double decompositionTime = decompositionEnd - decompositionStart;
            double gflops = (2 * M * N * K * 1e-9) / executionTime;
            int totalGPUs = size;
            int numberOfNodes = 1;

            char csvLine[100];
            sprintf(csvLine, "%s,%ld,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%lf,%lf,%lf\n", "cuBLASMp", M, N, K, Mb, Nb, dRow, dCol, numberOfNodes, totalGPUs, decompositionTime, executionTime, gflops);
            writeLineToFile(logfile, csvLine);
        }
    }

    CUDA_CHECK(cudaMemcpyAsync(localC, d_C, lldc * loc_n_c * sizeof(double), cudaMemcpyDeviceToHost, stream));

    CAL_CHECK(cal_stream_sync(calCommunicator, stream));
    CAL_CHECK(cal_comm_barrier(calCommunicator, stream));

    /* Gather results */
    decomposerC.gatherMatrix();

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

    if (rank == 0) {
        fclose(logfile);
    }

    return;
}
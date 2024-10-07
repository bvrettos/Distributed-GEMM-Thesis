#include "cuBLASMpWrapped.hpp"

template <typename scalar_t>
void cuBLASMpPreDistributedGemm(char TransA,  char TransB, const long long M, const long long N, const long long K,
  scalar_t alpha, scalar_t* A, const long long ldA, scalar_t* B, const long long ldB, scalar_t beta, scalar_t* C,
  const long long ldC, long int Mb, long int Nb, int numberOfRuns, bool logging, bool gatherResults)
{
    /* Assume that Matrices are always found on rank=0 */
    int ia = 1, ja = 1, ib = 1, jb = 1, ic = 1, jc = 1;
    FILE* logfile;

    /* 0. Handle input parameters */
    cublasOperation_t transOperationA, transOperationB;
    transOperationA = charToCublasTransOp(TransA);
    transOperationB = charToCublasTransOp(TransB);
    cudaDataType_t gemmDatatype;
    if (typeid(scalar_t) == typeid(float)) gemmDatatype = CUDA_R_32F;
    else if (typeid(scalar_t) == typeid(double)) gemmDatatype = CUDA_R_64F;
    int warmupRuns = 5;

    /* 1. Find rank and size of World */
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int totalGPUs = size;
    int numberOfNodes = getSlurmNumNodes();

    int dRow, dCol;
    calculateProcessGrid(&dRow, &dCol, size);

    if (logging) {
        /* Setup logfile */
        if (rank == 0) {
            std::string machineName = MACHINE_NAME;
            std::string filename = "DGEMM_execution_logs-PreDistributed_GEMM-" + machineName + "-cuBLASMp.csv";
            std::string header = "Algo,M,N,K,TileRows,TileColumns,dRow,dCol,TotalNodes,TotalGPUs,DecompositionTime,HostToDeviceTime,GatherTime,ExecutionTime,GFlops,DataLocation";
            logfile = createLogCsv(filename, header);
        }
    }

    /* 2. Find ID of localGPU - On Meluxina, each task gets 1 gpu, so each deviceID is 0 */
    int localDeviceID = 0;
    CUDA_CHECK(cudaSetDevice(localDeviceID));

    /* Row-Major Process Grid (2D - SUMMA) */
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
    pblasDecomposer decomposerA(M, K, Mb, Nb, MPI_COMM_WORLD);
    pblasDecomposer decomposerB(K, N, Mb, Nb, MPI_COMM_WORLD);
    pblasDecomposer decomposerC(M, N, Mb, Nb, MPI_COMM_WORLD);

    double decompositionEnd = MPI_Wtime();

    const int64_t llda = decomposerA.localRows;
    const int64_t loc_n_a = decomposerA.localColumns;

    const int64_t lldb = decomposerB.localRows;
    const int64_t loc_n_b = decomposerB.localColumns;

    const int64_t lldc = decomposerC.localRows;
    const int64_t loc_n_c = decomposerC.localColumns;

    /* Generate Matrix Data */
    double *localA, *localB, *localC;
    localA = (double*) malloc(sizeof(double) * llda * loc_n_a);
    localB = (double*) malloc(sizeof(double) * lldb * loc_n_b);
    localC = (double*) malloc(sizeof(double) * lldc * loc_n_c);

    MatrixInit(localA, llda, loc_n_a, 0);
    MatrixInit(localB, lldb, loc_n_b, 0);
    MatrixInit(localC, lldc, loc_n_c, 0);

    /* Host To Device. In metrics, if you want to show all-host or all-device, you need to either subtract or not this value */
    double memcpyStart = MPI_Wtime();
    CUDA_CHECK(cudaMallocAsync(&d_A, llda * loc_n_a * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_B, lldb * loc_n_b * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_C, lldc * loc_n_c * sizeof(double), stream));

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
        cublasMpMatrixDescriptorCreate(handle, M, K, Mb, Nb, 0, 0, llda, gemmDatatype, grid, &descA));
    CUBLAS_CHECK(
        cublasMpMatrixDescriptorCreate(handle, K, N, Mb, Nb, 0, 0, lldb, gemmDatatype, grid, &descB));
    CUBLAS_CHECK(
        cublasMpMatrixDescriptorCreate(handle, M, N, Mb, Nb, 0, 0, lldc, gemmDatatype, grid, &descC));

    /* 11. Calculate necessary memory size */
    CUBLAS_CHECK(cublasMpGemm_bufferSize(
        handle,
        transOperationA,
        transOperationB,
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
        gemmDatatype,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost));

    CUDA_CHECK(cudaMallocAsync(&d_work, workspaceInBytesOnDevice, stream));
    std::vector<int8_t> h_work(workspaceInBytesOnHost);

    CAL_CHECK(cal_stream_sync(calCommunicator, stream));
    CAL_CHECK(cal_comm_barrier(calCommunicator, stream));

    /* Simple warmup */
    for (int i = 0; i < warmupRuns; i++) {
        CUBLAS_CHECK(cublasMpGemm(
            handle,
            transOperationA,
            transOperationB,
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
            gemmDatatype,
            d_work,
            workspaceInBytesOnDevice,
            h_work.data(),
            workspaceInBytesOnHost));
    }   

    /* Re-copy original C matrix to GPUs */
    CUDA_CHECK(cudaMemcpyAsync(d_C, localC, lldc * loc_n_c * sizeof(double), cudaMemcpyHostToDevice, stream));
    
    CAL_CHECK(cal_stream_sync(calCommunicator, stream));
    CAL_CHECK(cal_comm_barrier(calCommunicator, stream));

    /* Actual runs */
    for (int i = 0; i < numberOfRuns; i++) {
        double executionStart = MPI_Wtime();
        CUBLAS_CHECK(cublasMpGemm(
            handle,
            transOperationA,
            transOperationB,
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
            gemmDatatype,
            d_work,
            workspaceInBytesOnDevice,
            h_work.data(),
            workspaceInBytesOnHost));

            
        /* Call barrier before stopping timer, all devices need to be available, not only root */
        CAL_CHECK(cal_stream_sync(calCommunicator, stream));
        CAL_CHECK(cal_comm_barrier(calCommunicator, stream));
        double executionEnd = MPI_Wtime();

        CUDA_CHECK(cudaMemcpyAsync(localC, d_C, lldc * loc_n_c * sizeof(double), cudaMemcpyDeviceToHost, stream));
        double gatherStart = MPI_Wtime();
        /* Gather results */
        if (gatherResults) {
            CAL_CHECK(cal_stream_sync(calCommunicator, stream));
            CAL_CHECK(cal_comm_barrier(calCommunicator, stream));
            decomposerC.gatherMatrix(0, C, localC);
        }

        double gatherEnd = MPI_Wtime();
        
        if (logging) {
            if (rank == 0) {
                double executionTime = executionEnd - executionStart;
                double decompositionTime = decompositionEnd - decompositionStart;
                double hostToDeviceTime = memcpyEnd - memcpyStart;
                double gatherTime = gatherEnd - gatherStart;
                double gflops = calculateGflops(M, N, K, executionTime);

                char csvLine[250];
                sprintf(csvLine, "%s,%ld,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%s\n", 
                "cuBLASMp", M, N, K, Mb, Nb, dRow, dCol, numberOfNodes, totalGPUs, decompositionTime, hostToDeviceTime, gatherTime, executionTime, gflops, "devices");
                writeLineToFile(logfile, csvLine);
            }
        }
    }

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

    if (logging) {
        if (rank == 0) {
            fclose(logfile);
        }
    }

    return;
}

template <typename scalar_t>
void cuBLASMpFullGemmOffload(char TransA,  char TransB, const long long M, const long long N, const long long K,
  scalar_t alpha, scalar_t* A, const long long ldA, scalar_t* B, const long long ldB, scalar_t beta, scalar_t* C,
  const long long ldC, long int Mb, long int Nb, int numberOfRuns, bool logging, bool gatherResults)
{
    /* Assume that Matrices are always found on rank=0 */
    int ia = 1, ja = 1, ib = 1, jb = 1, ic = 1, jc = 1;
    FILE* logfile;

    /* 0. Handle input parameters */
    cublasOperation_t transOperationA, transOperationB;
    transOperationA = charToCublasTransOp(TransA);
    transOperationB = charToCublasTransOp(TransB);
    cudaDataType_t gemmDatatype;
    if (typeid(scalar_t) == typeid(float)) gemmDatatype = CUDA_R_32F;
    else if (typeid(scalar_t) == typeid(double)) gemmDatatype = CUDA_R_64F;
    int warmupRuns = 5;

    /* 1. Find rank and size of World */
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int totalGPUs = size;
    int numberOfNodes = getSlurmNumNodes();

    int dRow, dCol;
    calculateProcessGrid(&dRow, &dCol, size);
    
    if (logging) {
        /* Setup logfile */
        if (rank == 0) {
            std::string machineName = MACHINE_NAME;
            std::string filename = "DGEMM_execution_logs-Full_GEMM_Offload-" + machineName + "-cuBLASMp.csv";
            std::string header = "Algo,M,N,K,TileRows,TileColumns,dRow,dCol,TotalNodes,TotalGPUs,DecompositionTime,HostToDeviceTime,GatherTime,ExecutionTime,GFlops,DataLocation";
            logfile = createLogCsv(filename, header);
        }
    }

    /* 2. Find ID of localGPU - On Meluxina, each task gets 1 gpu, so each deviceID is 0 */
    int localDeviceID = 0;
    CUDA_CHECK(cudaSetDevice(localDeviceID));

    /* Row-Major Process Grid (2D - SUMMA) */
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

    /* Decompose Matrices */
    double decompositionStart = MPI_Wtime();
    pblasDecomposer decomposerA(M, K, Mb, Nb, MPI_COMM_WORLD);
    pblasDecomposer decomposerB(K, N, Mb, Nb, MPI_COMM_WORLD);
    pblasDecomposer decomposerC(M, N, Mb, Nb, MPI_COMM_WORLD);

    double decompositionEnd = MPI_Wtime();

    const int64_t llda = decomposerA.localRows;
    const int64_t loc_n_a = decomposerA.localColumns;

    const int64_t lldb = decomposerB.localRows;
    const int64_t loc_n_b = decomposerB.localColumns;

    const int64_t lldc = decomposerC.localRows;
    const int64_t loc_n_c = decomposerC.localColumns;

    /* Scatter Matrix */
    double *localA, *localB, *localC;
    decomposerA.scatterMatrix(0, A, localA);
    decomposerB.scatterMatrix(0, B, localB);
    decomposerC.scatterMatrix(0, C, localC);

    /* Host To Device. In metrics, if you want to show all-host or all-device, you need to either subtract or not this value */
    double memcpyStart = MPI_Wtime();
    CUDA_CHECK(cudaMallocAsync(&d_A, llda * loc_n_a * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_B, lldb * loc_n_b * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_C, lldc * loc_n_c * sizeof(double), stream));

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
        cublasMpMatrixDescriptorCreate(handle, M, K, Mb, Nb, 0, 0, llda, gemmDatatype, grid, &descA));
    CUBLAS_CHECK(
        cublasMpMatrixDescriptorCreate(handle, K, N, Mb, Nb, 0, 0, lldb, gemmDatatype, grid, &descB));
    CUBLAS_CHECK(
        cublasMpMatrixDescriptorCreate(handle, M, N, Mb, Nb, 0, 0, lldc, gemmDatatype, grid, &descC));

    /* 11. Calculate necessary memory size */
    CUBLAS_CHECK(cublasMpGemm_bufferSize(
        handle,
        transOperationA,
        transOperationB,
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
        gemmDatatype,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost));

    CUDA_CHECK(cudaMallocAsync(&d_work, workspaceInBytesOnDevice, stream));
    std::vector<int8_t> h_work(workspaceInBytesOnHost);

    CAL_CHECK(cal_stream_sync(calCommunicator, stream));
    CAL_CHECK(cal_comm_barrier(calCommunicator, stream));

    /* Simple warmup */
    for (int i = 0; i < warmupRuns; i++) {
        CUBLAS_CHECK(cublasMpGemm(
            handle,
            transOperationA,
            transOperationB,
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
            gemmDatatype,
            d_work,
            workspaceInBytesOnDevice,
            h_work.data(),
            workspaceInBytesOnHost));
    }

    /* Re-copy original C matrix to GPUs */
    CUDA_CHECK(cudaMemcpyAsync(d_C, localC, lldc * loc_n_c * sizeof(double), cudaMemcpyHostToDevice, stream));
    
    CAL_CHECK(cal_stream_sync(calCommunicator, stream));
    CAL_CHECK(cal_comm_barrier(calCommunicator, stream));

    /* Actual runs */
    for (int i = 0; i < numberOfRuns; i++) {
        double executionStart = MPI_Wtime();
        CUBLAS_CHECK(cublasMpGemm(
            handle,
            transOperationA,
            transOperationB,
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
            gemmDatatype,
            d_work,
            workspaceInBytesOnDevice,
            h_work.data(),
            workspaceInBytesOnHost));
        CAL_CHECK(cal_stream_sync(calCommunicator, stream));
        CAL_CHECK(cal_comm_barrier(calCommunicator, stream));

        /* Call barrier before stopping timer, all devices need to be available, not only root */
        double executionEnd = MPI_Wtime();

        double gatherStart = MPI_Wtime();
        /* Gather results */
        if (gatherResults) {
            CUDA_CHECK(cudaMemcpyAsync(localC, d_C, lldc * loc_n_c * sizeof(double), cudaMemcpyDeviceToHost, stream));    
            CAL_CHECK(cal_stream_sync(calCommunicator, stream));
            CAL_CHECK(cal_comm_barrier(calCommunicator, stream));
            decomposerC.gatherMatrix(0, C, localC);
        }
        double gatherEnd = MPI_Wtime();
        
        if (logging) {
            if (rank == 0) {
                double executionTime = executionEnd - executionStart;
                double decompositionTime = decompositionEnd - decompositionStart;
                double hostToDeviceTime = memcpyEnd - memcpyStart;
                double gatherTime = gatherEnd - gatherStart;
                double gflops = calculateGflops(M, N, K, executionTime);

                char csvLine[200];
                sprintf(csvLine, "%s,%ld,%ld,%ld,%ld,%ld,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%lf,%s\n", 
                "cuBLASMp", M, N, K, Mb, Nb, dRow, dCol, numberOfNodes, totalGPUs, decompositionTime, hostToDeviceTime, gatherTime, executionTime, gflops, "devices");
                writeLineToFile(logfile, csvLine);
            }
        }
    }

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

    if (logging) {
        if (rank == 0) {
            fclose(logfile);
        }
    }

    return;
}

template <typename scalar_t>
void cuBLASMpGEMMWrap(char TransA,  char TransB, const long long M, const long long N, const long long K,
  scalar_t alpha, scalar_t* A, const long long ldA, scalar_t* B, const long long ldB, scalar_t beta, scalar_t* C,
  const long long ldC, long int Mb, long int Nb)
{
    /* Assume that Matrices are always found on rank=0 */
    int ia = 1, ja = 1, ib = 1, jb = 1, ic = 1, jc = 1;
    
    /* 0. Handle input parameters */
    cublasOperation_t transOperationA, transOperationB;
    transOperationA = charToCublasTransOp(TransA);
    transOperationB = charToCublasTransOp(TransB);
    cudaDataType_t gemmDatatype;
    if (typeid(scalar_t) == typeid(float)) gemmDatatype = CUDA_R_32F;
    else if (typeid(scalar_t) == typeid(double)) gemmDatatype = CUDA_R_64F;

    /* 1. Find rank and size of World */
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dRow, dCol;
    calculateProcessGrid(&dRow, &dCol, size);

    /* 2. Find ID of localGPU - On Meluxina, each task gets 1 gpu, so each deviceID is 0 */
    int localDeviceID = 0;
    CUDA_CHECK(cudaSetDevice(localDeviceID));

    /* Row-Major Process Grid (2D - SUMMA) */
    int localRow = rank / dCol;
    int localCol = rank % dCol;

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
    pblasDecomposer decomposerA(M, K, Mb, Nb, MPI_COMM_WORLD);
    pblasDecomposer decomposerB(K, N, Mb, Nb, MPI_COMM_WORLD);
    pblasDecomposer decomposerC(M, N, Mb, Nb, MPI_COMM_WORLD);

    const int64_t llda = decomposerA.localRows;
    const int64_t loc_n_a = decomposerA.localColumns;

    const int64_t lldb = decomposerB.localRows;
    const int64_t loc_n_b = decomposerB.localColumns;

    const int64_t lldc = decomposerC.localRows;
    const int64_t loc_n_c = decomposerC.localColumns;

    double *localA, *localB, *localC;
    localA = (double*) malloc(sizeof(double) * llda * loc_n_a);
    localB = (double*) malloc(sizeof(double) * lldb * loc_n_b);
    localC = (double*) malloc(sizeof(double) * lldc * loc_n_c);

    double decompStart = MPI_Wtime();
    decomposerA.scatterMatrix(0, A, localA);
    decomposerB.scatterMatrix(0, B, localB);
    decomposerC.scatterMatrix(0, C, localC);
    double decompEnd = MPI_Wtime();

    double decomp = decompEnd-decompStart;
    if (rank == 0)
        printf("Decomp Time: %lf\n", decomp);

    /* Host To Device. In metrics, if you want to show all-host or all-device, you need to either subtract or not this value */
    CUDA_CHECK(cudaMallocAsync(&d_A, llda * loc_n_a * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_B, lldb * loc_n_b * sizeof(double), stream));
    CUDA_CHECK(cudaMallocAsync(&d_C, lldc * loc_n_c * sizeof(double), stream));

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
        cublasMpMatrixDescriptorCreate(handle, M, K, Mb, Nb, 0, 0, llda, gemmDatatype, grid, &descA));
    CUBLAS_CHECK(
        cublasMpMatrixDescriptorCreate(handle, K, N, Mb, Nb, 0, 0, lldb, gemmDatatype, grid, &descB));
    CUBLAS_CHECK(
        cublasMpMatrixDescriptorCreate(handle, M, N, Mb, Nb, 0, 0, lldc, gemmDatatype, grid, &descC));

    /* 11. Calculate necessary memory size */
    CUBLAS_CHECK(cublasMpGemm_bufferSize(
        handle,
        transOperationA,
        transOperationB,
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
        gemmDatatype,
        &workspaceInBytesOnDevice,
        &workspaceInBytesOnHost));

    CUDA_CHECK(cudaMallocAsync(&d_work, workspaceInBytesOnDevice, stream));
    std::vector<int8_t> h_work(workspaceInBytesOnHost);

    CAL_CHECK(cal_stream_sync(calCommunicator, stream));
    CAL_CHECK(cal_comm_barrier(calCommunicator, stream));
    
    double executionBefore = MPI_Wtime();
    CUBLAS_CHECK(cublasMpGemm(
        handle,
        transOperationA,
        transOperationB,
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
        gemmDatatype,
        d_work,
        workspaceInBytesOnDevice,
        h_work.data(),
        workspaceInBytesOnHost));

    CAL_CHECK(cal_stream_sync(calCommunicator, stream));
    CAL_CHECK(cal_comm_barrier(calCommunicator, stream));
    double executionAfter = MPI_Wtime();
    if (rank == 0)
        printf("ExecutionTime: %lf\n", executionAfter - executionBefore);
    CUDA_CHECK(cudaMemcpyAsync(localC, d_C, lldc * loc_n_c * sizeof(double), cudaMemcpyDeviceToHost, stream));
    CAL_CHECK(cal_stream_sync(calCommunicator, stream));
    CAL_CHECK(cal_comm_barrier(calCommunicator, stream));

    double gatherStart = MPI_Wtime();
    decomposerC.gatherMatrix(0, C, localC);
    double gatherEnd = MPI_Wtime();
    double gather = gatherEnd-gatherStart;
    if (rank == 0)
        printf("Gather: %lf\n", gather);

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

    return;
}

template void cuBLASMpGEMMWrap<double>(char TransA,  char TransB, const long long M, const long long N, const long long K,
  double alpha, double* A, const long long ldA, double* B, const long long ldB, double beta, double* C,
  const long long ldC, long int Mb, long int Nb);

template void cuBLASMpPreDistributedGemm<double>(char TransA,  char TransB, const long long M, const long long N, const long long K,
  double alpha, double* A, const long long ldA, double* B, const long long ldB, double beta, double* C,
  const long long ldC, long int Mb, long int Nb, int numberOfRuns, bool logging, bool gatherResults);

template void cuBLASMpFullGemmOffload<double>(char TransA,  char TransB, const long long M, const long long N, const long long K,
  double alpha, double* A, const long long ldA, double* B, const long long ldB, double beta, double* C,
  const long long ldC, long int Mb, long int Nb, int numberOfRuns, bool logging, bool gatherResults);
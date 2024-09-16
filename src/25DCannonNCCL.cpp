#include <25DCannon.hpp>

// #define STREAM_HARD_CAP 8 // maximum computation and communication overlap 
#define GPU_MAX_MEMORY_USAGE 0.95 // Percentage (currently 95%)

int truncatedDivisionRemainer(int a, int b)
{
    return ((a % b) + b) % b;
}

/* 
    Calculate how many streams each GPU can handle. 
    If all tiles fit in memory for every execution, maximum overlap can happen.
    Amount of available overlap is shown via how many CUDA streams we create. We
    can create infinite streams but most modern GPUs can handle 32 maximum-hardware wise. The rest will be
    sequential. Should not utilize 100% of GPU, set a hard-cap for memory usage.

    In the end, I find out that this is completely unecessary. Cannon's algorithm cannot have communication/computation
    overlap since there are as many tasks as there are workers. Comm/Comp overlap needs tiling, 2.5D SUMMA is much better for this one.
*/

void Summa25Decomposer::communicationComputationOverlapInit()
{
    int kernelsToCompute = pc3;
    size_t singleExecutionMemoryRequirements; // sizeof(localA) + sizeof(localB) + sizeof(localC)
    size_t tileMemoryRequirements = sizeof(double) * (localM*localK);
    singleExecutionMemoryRequirements = sizeof(double) * (localM*localK + localK*localN + localM*localN);

    size_t totalExecutionMemoryRequirements = tileMemoryRequirements*(2*2 + 1); // 1 Tile for C and 2*2 tiles for A and B (to have concurrent streams on communication)
    long long freeMemory, maxMemory;
    getGPUMemoryInfo(&freeMemory, &maxMemory, 0);

    freeMemory *= GPU_MAX_MEMORY_USAGE;

    #ifdef DEBUG
        if (rank == 0) {
            printf("Kernels to Compute: %d, singleExecutionMemory(in elements): %lld, maxMemoryAllowed(in elements): %lld\n", kernelsToCompute, singleExecutionMemoryRequirements/8, freeMemory/8);
        }
    #endif

    if (freeMemory >= totalExecutionMemoryRequirements) {
        /* Every kernel fits in memory, create as many streams as the kernels. */
        printf("Problem fits all computations in memory\n");
    }
    else {
        printf("Problem does not fit in GPUs, cannot utilize NCCL. Use MPI Version.\n");
        MPI_Abort(MPI_COMM_WORLD, -5);
    }
    activeStreams = kernelsToCompute;

    cudaSetDevice(0);
    streams = new cudaStream_t[activeStreams]; // Allocate array for streams
    events = new cudaEvent_t[activeStreams]; // Allocate array for events 
    workspaceA = new double*[2];
    workspaceB = new double*[2];

    /* 
       Initialze streams and workload memory for each one. C does not need multiple allocations
       only A, B which are dependencies for multiplication.
    */
    for (int i = 0; i < activeStreams; i++) {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
    }

    for (int i = 0; i < 2; i++) {
        cudaMallocAsync((void**)&workspaceA[i], sizeof(double) * localM * localK, streams[i]);
        cudaMallocAsync((void**)&workspaceB[i], sizeof(double) * localN * localK, streams[i]);
    }

    CUDA_CHECK(cudaStreamCreate(&cublasStream));
    // CUBLAS_CHECK(cublasSetStream(cublasContext, cublasStream)); // Stream set for initial execution

    cudaStreamSynchronize(streams[0]);

    if (rank == 0)
        printf("Active Streams per Device: %d\n", activeStreams);

    initializeNCCL();
    
    return;
}

void Summa25Decomposer::broadcastToStackNCCL(double* aTile, double* bTile)
{   
    /* All NCCL calls are synced */
    if (rank == 0) printf("Broadcasting A and B across stack communicator\n");
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclBroadcast(aTile, aTile, localM * localK, ncclDouble, 0, ncclCommonStackCommunicator, NULL));
    NCCL_CHECK(ncclBroadcast(bTile, bTile, localN * localK, ncclDouble, 0, ncclCommonStackCommunicator, NULL));
    NCCL_CHECK(ncclGroupEnd());

    broadcastComplete = true;

    return;
}

void Summa25Decomposer::initializeNCCL()
{
    /* Create the commonStackCommunicator for NCCL */
    if (rank == 0) ncclGetUniqueId(&uniqueId);
    MPI_CHECK(MPI_Bcast(&uniqueId, sizeof(uniqueId), MPI_BYTE, 0, MPI_COMM_WORLD));
    NCCL_CHECK(ncclCommInitRank(&ncclCommunicator, size, uniqueId, rank));
    NCCL_CHECK(ncclCommSplit(ncclCommunicator, (processColumn + processRow*dCol), rank, &ncclCommonStackCommunicator, NULL));

    return;
}

void Summa25Decomposer::multiplyNCCL(char TransA, char TransB, double* A, double* B, double* C, double alpha, double beta)
{
    if (!communicationComputationOverlap) {
        if (rank == 0) printf("Call multiply, not multiply NNCL\n");
        MPI_Abort(MPI_COMM_WORLD, -6);
    }

    /* Skip this if c == 1 or have already completed a broadcast */
    if (c > 1) {
        if (!broadcastComplete) broadcastToStackNCCL(localA, localB);
    }

    cudaStream_t* currentCommunicationStream = &streams[0];
    cudaEvent_t* currentEvent = &events[0];

    /* First Shift for Aij. Find out who is going to give you the Aij block. Then, find out who is going to receive Aij. */
    int s = truncatedDivisionRemainer((processColumn - processRow + processStack*pc3), pc);
    int receiverRankForA = processRow*dCol + processStack*dRow*dCol + s;
    int senderRankForA = -1;

    int s_dot = truncatedDivisionRemainer((processRow - processColumn + processStack*pc3), pc);
    int receiverRankForB = s_dot*dCol + processStack*dRow*dCol + processColumn;
    int senderRankForB = -1;

    /*  Trade ranks: Send your rank to whoever is going to receive from you and receive the rank of the sender. This is
        necessary due to NCCL not having dynamic point-to-point communication. Basically, you cannot use MPI_ANY_SOURCE
        with NCCL.
    */

    MPI_CHECK(MPI_Sendrecv(&rank, 1, MPI_INT, receiverRankForA, 0, &senderRankForA, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    MPI_CHECK(MPI_Sendrecv(&rank, 1, MPI_INT, receiverRankForB, 1, &senderRankForB, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    MPI_Barrier(MPI_COMM_WORLD);

    /* Trade Tiles */
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(localA, localM * localK, ncclDouble, receiverRankForA, ncclCommunicator, *currentCommunicationStream));
    NCCL_CHECK(ncclRecv(workspaceA[0], localM * localK, ncclDouble, senderRankForA, ncclCommunicator, *currentCommunicationStream));
    NCCL_CHECK(ncclGroupEnd());

    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(localB, localN * localK, ncclDouble, receiverRankForB, ncclCommunicator, *currentCommunicationStream));
    NCCL_CHECK(ncclRecv(workspaceB[0], localN * localK, ncclDouble, senderRankForB, ncclCommunicator, *currentCommunicationStream));
    NCCL_CHECK(ncclGroupEnd());

    /*
        Record event and force cublasStream to wait until transfer has been complete, continue with communication.
        As soon as this event is complete, you can free up this stream and let someone else use it. 
    */
    CUDA_CHECK(cudaEventRecord(*currentEvent, *currentCommunicationStream));

    /* Wait for transfer stream */
    CUDA_CHECK(cudaStreamWaitEvent(cublasStream, *currentEvent));

    /* If your processStack is 0, meaning that you have the original C matrix, you need to add + beta*Cij to the sum. */
    double tempBeta = (processStack == 0) ? beta : 0.00000; // By setting beta=0, cuBLAS ignores C matrix data. 
    cudaDeviceSynchronize();
    CUBLAS_CHECK(cublasDgemm(cublasContext, charToCublasTransOp(TransA), charToCublasTransOp(TransB), 
        localM, localN, localK, &alpha, workspaceA[0], llda, workspaceB[0], lldb, &tempBeta, localC, lldc));

    /* If c = p^(1/3), the algorithm should end here since we basically run 3D. If not, then we need to compute more tiles */
    for (int rotationCount = 1; rotationCount < pc3; rotationCount++) {
        cudaEvent_t* previousEvent = currentEvent;
        int computationIndex = rotationCount % 2; // 0 or 1
        int previousComputationIndex = (computationIndex == 1) ? 0 : 1;

        currentCommunicationStream = &streams[rotationCount];
        currentEvent = &events[rotationCount];
        
        s = (processColumn + rotationCount) % pc;
        s_dot = (processRow + rotationCount) % pc;

        /* Shift tiles once and trade ranks to achieve communication with NCCL. */
        receiverRankForA = processRow*dCol + processStack*dRow*dCol + s;
        receiverRankForB = s_dot*dCol + processStack*dRow*dCol + processColumn;
        MPI_CHECK(MPI_Sendrecv(&rank, 1, MPI_INT, receiverRankForA, 0, &senderRankForA, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        MPI_CHECK(MPI_Sendrecv(&rank, 1, MPI_INT, receiverRankForB, 1, &senderRankForB, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

        CUDA_CHECK(cudaStreamWaitEvent(*currentCommunicationStream, *previousEvent));
        
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclSend(workspaceA[previousComputationIndex], localM * localK, ncclDouble, receiverRankForA, ncclCommunicator, *currentCommunicationStream));
        NCCL_CHECK(ncclRecv(workspaceA[computationIndex], localM * localK, ncclDouble, senderRankForA, ncclCommunicator, *currentCommunicationStream));
        NCCL_CHECK(ncclGroupEnd());

        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclSend(workspaceB[previousComputationIndex], localN * localK, ncclDouble, receiverRankForB, ncclCommunicator, *currentCommunicationStream));
        NCCL_CHECK(ncclRecv(workspaceB[computationIndex], localN * localK, ncclDouble, senderRankForB, ncclCommunicator, *currentCommunicationStream));
        NCCL_CHECK(ncclGroupEnd());
        
        CUDA_CHECK(cudaEventRecord(*currentEvent, *currentCommunicationStream));  
        CUDA_CHECK(cudaStreamWaitEvent(cublasStream, *currentEvent));

        double tempBeta = 1.00;
        CUBLAS_CHECK(cublasDgemm(cublasContext, charToCublasTransOp(TransA), charToCublasTransOp(TransB), 
            localM, localN, localK, &alpha, workspaceA[computationIndex], llda, workspaceB[computationIndex], lldb, &tempBeta, localC, lldc));
    }

    /* Synchronize device and reduce results*/
    cudaStreamSynchronize(cublasStream);
    reduceResultNCCL(localC);

    /* Reset broadcastComplete for future runs */
    broadcastComplete = false;

    return;
}

void Summa25Decomposer::serializedMultiplyNCCL(char TransA, char TransB, double* A, double* B, double* C, double alpha, double beta)
{
    if (!communicationComputationOverlap) {
        if (rank == 0) printf("Call multiply, not multiply NNCL\n");
        MPI_Abort(MPI_COMM_WORLD, -6);
    }

    /* Skip this if c == 1 or have already completed a broadcast */
    if (c > 1) {
        if (!broadcastComplete) broadcastToStackNCCL(localA, localB);
    }

    /* First Shift for Aij. Find out who is going to give you the Aij block. Then, find out who is going to receive Aij. */
    int s = truncatedDivisionRemainer((processColumn - processRow + processStack*pc3), pc);
    int receiverRankForA = processRow*dCol + processStack*dRow*dCol + s;
    int senderRankForA = -1;

    int s_dot = truncatedDivisionRemainer((processRow - processColumn + processStack*pc3), pc);
    int receiverRankForB = s_dot*dCol + processStack*dRow*dCol + processColumn;
    int senderRankForB = -1;

    /*  Trade ranks: Send your rank to whoever is going to receive from you and receive the rank of the sender. This is
        necessary due to NCCL not having dynamic point-to-point communication. Basically, you cannot use MPI_ANY_SOURCE
        with NCCL.
    */
    
    MPI_CHECK(MPI_Sendrecv(&rank, 1, MPI_INT, receiverRankForA, 0, &senderRankForA, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    MPI_CHECK(MPI_Sendrecv(&rank, 1, MPI_INT, receiverRankForB, 1, &senderRankForB, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    MPI_Barrier(MPI_COMM_WORLD);
    
    /* Trade Tiles */
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(localA, localM * localK, ncclDouble, receiverRankForA, ncclCommunicator, NULL));
    NCCL_CHECK(ncclRecv(workspaceA[0], localM * localK, ncclDouble, senderRankForA, ncclCommunicator, NULL));
    NCCL_CHECK(ncclGroupEnd());

    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclSend(localB, localN * localK, ncclDouble, receiverRankForB, ncclCommunicator, NULL));
    NCCL_CHECK(ncclRecv(workspaceB[0], localN * localK, ncclDouble, senderRankForB, ncclCommunicator, NULL));
    NCCL_CHECK(ncclGroupEnd());

    /* If your processStack is 0, meaning that you have the original C matrix, you need to add + beta*Cij to the sum. */
    double tempBeta = (processStack == 0) ? beta : 0.00000; // By setting beta=0, cuBLAS ignores C matrix data. 
    
    CUBLAS_CHECK(cublasDgemm(cublasContext, charToCublasTransOp(TransA), charToCublasTransOp(TransB), 
        localM, localN, localK, &alpha, workspaceA[0], llda, workspaceB[0], lldb, &tempBeta, localC, lldc));

    /* If c = p^(1/3), the algorithm should end here since we basically run 3D SUMMA. If not, then we need to compute more tiles */
    for (int rotationCount = 1; rotationCount < pc3; rotationCount++) {
        int computationIndex = rotationCount % 2; // 0 or 1
        int previousComputationIndex = (computationIndex == 1) ? 0 : 1;
        
        s = (processColumn + rotationCount) % pc;
        s_dot = (processRow + rotationCount) % pc;

        /* Shift tiles once and trade ranks to achieve communication with NCCL. */
        receiverRankForA = processRow*dCol + processStack*dRow*dCol + s;
        receiverRankForB = s_dot*dCol + processStack*dRow*dCol + processColumn;
        MPI_CHECK(MPI_Sendrecv(&rank, 1, MPI_INT, receiverRankForA, 0, &senderRankForA, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        MPI_CHECK(MPI_Sendrecv(&rank, 1, MPI_INT, receiverRankForB, 1, &senderRankForB, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        MPI_Barrier(MPI_COMM_WORLD);

        
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclSend(workspaceA[previousComputationIndex], localM * localK, ncclDouble, receiverRankForA, ncclCommunicator, NULL));
        NCCL_CHECK(ncclRecv(workspaceA[computationIndex], localM * localK, ncclDouble, senderRankForA, ncclCommunicator, NULL));
        NCCL_CHECK(ncclGroupEnd());

        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclSend(workspaceB[previousComputationIndex], localN * localK, ncclDouble, receiverRankForB, ncclCommunicator, NULL));
        NCCL_CHECK(ncclRecv(workspaceB[computationIndex], localN * localK, ncclDouble, senderRankForB, ncclCommunicator, NULL));
        NCCL_CHECK(ncclGroupEnd());


        double tempBeta = 1.00;
        CUBLAS_CHECK(cublasDgemm(cublasContext, charToCublasTransOp(TransA), charToCublasTransOp(TransB), 
            localM, localN, localK, &alpha, workspaceA[computationIndex], llda, workspaceB[computationIndex], lldb, &tempBeta, localC, lldc));
    }

    /* Synchronize device and reduce results*/
    cudaDeviceSynchronize();
    reduceResultNCCL(localC);

    /* Reset broadcastComplete for future runs */
    broadcastComplete = false;

    return;

}

void Summa25Decomposer::reduceResultNCCL(double* cTile)
{
    NCCL_CHECK(ncclReduce(
        cTile,
        cTile,
        localM*localN,
        ncclDouble,
        ncclSum,
        0,
        ncclCommonStackCommunicator,
        NULL
    ));

    return;
}

void preDistributedCannon25NCCL(char TransA, char TransB, long long M, long long N, long long K, 
    double alpha, double* A, long long lda, double* B, long long ldb, double beta, double* C, long long ldc, int dRow, int dCol, int c,
    int numberOfRuns, bool logging, bool gatherResults)
{
    int rank, size;
    int numberOfNodes = getSlurmNumNodes();
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Logging */
    FILE* logfile;
    if (logging) {
        if (rank == 0) {
            std::string machineName = MACHINE_NAME;
            std::string filename = "DGEMM_execution_logs-PreDistributed_GEMM-" + machineName + "-2.5DCANNON-cuBLAS-NCCL.csv";
            std::string header = "Algo,M,N,K,TotalNodes,TotalGPUs,dRow,dCol,StackSize,DecompositionTime,ExecutionTime,GatherTime,GFlops,DataLocation";
            logfile = createLogCsv(filename, header);
        }
    }

    /* Create Decomposer */
    double beforeDecomp = MPI_Wtime();
    Summa25Decomposer decomposer(M, N, K, dRow, dCol, c, true);
    double afterDecomp = MPI_Wtime();
    long long localM = decomposer.localM;
    long long localN = decomposer.localN;
    long long localK = decomposer.localK;
    /* PreDistributed -> Do not call Decompose2D, just generate data on processStack == 0*/
    int processStack = decomposer.processStack;

    if (processStack == 0) {
        MatrixInit(decomposer.localA, localM * localK, 0);
        MatrixInit(decomposer.localB, localK * localN, 0);
        MatrixInit(decomposer.localC, localM * localN, 0);
    }

    /* Warmup runs */
    for (int i = 0; i < 10; i++) {
        /* Warmup...*/
    }

    /* Actual runs */
    for (int i = 0; i < numberOfRuns; i++) {
        double beforeExecution = MPI_Wtime();
        // decomposer.multiplyNCCL(TransA, TransB, A, B, C, alpha, beta);
        decomposer.serializedMultiplyNCCL('N', 'N', A, B, C, alpha, beta);
        // cudaDeviceSynchronize();
        // MPI_Barrier(MPI_COMM_WORLD);
        double afterExecution = MPI_Wtime();

        double beforeGather = MPI_Wtime();
        if (gatherResults) {
            decomposer.reorderMatrix(0, C, decomposer.localC);
        }

        double afterGather = MPI_Wtime();

        if (logging) {
            if (rank == 0) {
                double decompTime = afterDecomp - beforeDecomp;
                double executionTime = afterExecution - beforeExecution;
                double gatherTime = afterGather - beforeGather;
                double gflops = calculateGflops(M, N, K, executionTime);
                char csvLine[300];
                sprintf(csvLine, "%s,%lld,%lld,%lld,%d,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%s\n",
                                "2.5D CANNON cuBLAS NCCL", M, N, K, numberOfNodes, size, dRow, dCol, c, decompTime, executionTime, gatherTime, gflops, "devices");
                writeLineToFile(logfile, csvLine);
            }
        }
    }

    if (logging) {
        if (rank == 0) {
            fclose(logfile);
        }
    }

    return;
}
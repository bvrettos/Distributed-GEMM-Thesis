#include <25DSumma.hpp>
// #define DEBUG

int truncatedDivisionRemainer(int a, int b)
{
    return ((a % b) + b) % b;
}

Summa25Decomposer::Summa25Decomposer(long long M, long long N, long long K, int dRow, int dCol, int c) : M(M), N(N), K(K), dRow(dRow), dCol(dCol), c(c)
{
    /* Check if MPI is initialized */
    if (M != N || M != K) {
        printf("2.5D Summa currently supports only Square Matrices (M = N = K)\n");
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (dRow != dCol) {
        printf("2.5D Summa currently supports only Square Process Grids (dRow = dCol)\n");
        MPI_Abort(MPI_COMM_WORLD, -2);
    }

    if (dRow * dCol * c != size) {
        printf("Process Dimensions not matching with size, change communicator size\n");
        MPI_Abort(MPI_COMM_WORLD, -2);
    }

    /* 
       All checks passed???
       1. Assume that matrices are square 
       2. Process Grid is also square
       3. For k = 0, each process should own a M/(sqrt(p/c)) tile of A, B and C
    */

    /* Calculate process (i,j,k) for each rank (rank = j + i*dCol + k*dRow*dCol)*/
    processStack = rank/(dRow*dCol);        //k
    processRow = (rank/dCol)%dRow;          //i
    processColumn = (rank%dCol)%dRow;       //j

    pc3 = std::sqrt(size/(c*c*c));
    pc = std::sqrt(size/c);

    localM = M/dRow;
    localN = N/dCol;
    localK = K/dRow;

    llda = localM;
    lldb = localK;
    lldc = localM;

    /* Allocate local pointers */
    // localA = (double*) malloc(sizeof(double) * localM * localK);
    // localB = (double*) malloc(sizeof(double) * localN * localK);
    // localC = (double*) malloc(sizeof(double) * localM * localN);
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc((void**)&localA, sizeof(double) * localM * localK));
    CUDA_CHECK(cudaMalloc((void**)&localB, sizeof(double) * localK * localN));
    CUDA_CHECK(cudaMalloc((void**)&localC, sizeof(double) * localM * localN));

    #ifdef DEBUG
        printf("Rank: %d, (i,j,k) = (%d,%d,%d)\n", rank, processRow, processColumn, processStack);
    #endif

    createCommunicators();
    broadcastComplete = false; // This changes to true when original A and B matrices have been sent to other stack processes

    /* Create cuBLAS context */
    CUBLAS_CHECK(cublasCreate(&cublasContext));

    #ifdef DEBUG
        if (rank == 0) {
            printf("Successfully decomposed original matrices to 2D SUMMA\n");
        }
        printf("2.5D SUMMA Decomposer Initialized\n");
    #endif

    return;
}

/* 
    Distribute original A,B,C matrices on k = 0 grid
    For now, we assume that A,B,C are square with M = N = K. 
    I could do it with scatters but for future-proofing reasons it is going to be
    implemented with point-2-point communication
*/
void Summa25Decomposer::decompose2D(int senderRank, double* A, double* B, double* C)
{
    /* 
        Original SUMMA decomposition should only happen on the grid with k = 0. Don't know if 
        straight up returning is the best method
    */
    if (processStack != 0)
        return;
    
    int sendCount = 0, receiveCount = 3;
    if (rank == senderRank) {
        sendCount = 3*(dRow*dCol - 1);
        receiveCount = 0;
    }

    MPI_Request* sendRequests;
    if (sendCount > 0) {
        sendRequests = new MPI_Request[sendCount];
    }
    MPI_Request* receiveRequests;
    if (receiveCount > 0) {
        receiveRequests = new MPI_Request[receiveCount];
    }

    int sendIndex = 0, receiveIndex = 0;

    for (int i = 0; i < dRow; i++) {
        for (int j = 0; j < dCol; j++) {
            int receiverRank = i*dCol + j;

            if (rank == senderRank) {
                if (rank == receiverRank) {
                    copyBlock(localM, localK, &A[localM*(j*M + i)], localA, M, localM);
                    copyBlock(localK, localN, &B[localK*(j*K + i)], localB, K, localK);
                    copyBlock(localM, localN, &C[localM*(j*M + i)], localC, M, localM);
                }
                else {
                    transferBlock(localM ,localK, &A[localM*(j*M + i)], M, receiverRank, 0, &sendRequests[sendIndex++]);
                    transferBlock(localK ,localN, &B[localK*(j*K + i)], K, receiverRank, 0, &sendRequests[sendIndex++]);
                    transferBlock(localM ,localN, &C[localM*(j*M + i)], M, receiverRank, 0, &sendRequests[sendIndex++]);
                }
            }

            if ((rank == receiverRank) && (rank != senderRank)) {
                receiveBlock(localM, localK, localA, localM, senderRank, 0, &receiveRequests[receiveIndex++]);
                receiveBlock(localK, localN, localB, localK, senderRank, 0, &receiveRequests[receiveIndex++]);
                receiveBlock(localM, localN, localC, localM, senderRank, 0, &receiveRequests[receiveIndex++]);
            }
        }
    }
    
    if (receiveIndex > 0) {
        MPI_Waitall(receiveIndex, receiveRequests, MPI_STATUSES_IGNORE);
        delete[] receiveRequests;
    }

    if (sendIndex > 0) {
        MPI_Waitall(sendIndex, sendRequests, MPI_STATUSES_IGNORE);
        delete[] sendRequests;
    }

    return;
}

void Summa25Decomposer::createCommunicators()
{
    /* Create communicators */
    MPI_Comm_split(MPI_COMM_WORLD, processStack, rank, &commonGridCommunicator); /* Puts ranks of the same stack in a 2D-Grid */
    MPI_Comm_split(MPI_COMM_WORLD, (processColumn + processRow*dCol), rank, &commonStackCommunicator); /* Puts ranks of the same (i,j) on the same stack */
    MPI_Comm_split(commonGridCommunicator, processRow, rank, &commonRowCommunicator); /* Puts rank of the same row of the 2D-Grid on the same communicator */
    MPI_Comm_split(commonGridCommunicator, processColumn, rank, &commonColumnCommunicator); /* Puts rank of the same column of the 2D-Grid on the same communicator */

    MPI_Comm_rank(commonStackCommunicator, &commonStackRank);

    #ifdef DEBUG
        printf("Rank: %d, commonStackRank: %d\n", rank, commonStackRank);
        printf("Communicators Initialized Successfully\n");
    #endif

    return;
}

void Summa25Decomposer::multiply(char TransA, char TransB, double* A, double* B, double* C, double alpha, double beta)
{   
    MPI_Request *broadcastRequests;
    int broadcastIndex = 0;

    /* Skip this if c == 1 */
    if ((c > 1) && !broadcastComplete) {
        broadcastRequests = new MPI_Request[2];
        /* Go from 2D Decomposition to 2.5D -> Broadcast Aij and Bij on the common Stack Communicator */
        MPI_Ibcast(localA, localM * localK, MPI_DOUBLE, 0, commonStackCommunicator, &broadcastRequests[broadcastIndex++]);
        MPI_Ibcast(localB, localN * localK, MPI_DOUBLE, 0, commonStackCommunicator, &broadcastRequests[broadcastIndex++]);

        /* Wait for broadcast, if you are the root, you can continue */
        if (processStack != 0) {
            MPI_Waitall(broadcastIndex, broadcastRequests, MPI_STATUSES_IGNORE);
        }

        delete[] broadcastRequests; // Do not need this anymore 
        broadcastComplete = true;
    }

    double beforeFirstShift = MPI_Wtime();
    /* Find out who is going to receive Aij. Because we are going to be using MPI_ANY_SOURCE, we need to set tags. Aij is assigned tag=0 and Bij is assigned tag=1 */
    int s = truncatedDivisionRemainer((processColumn - processRow + processStack*pc3), pc);
    int receiverRankForA = processRow*dCol + processStack*dRow*dCol + s;
    MPI_Sendrecv_replace(localA, localM * localK, MPI_DOUBLE, receiverRankForA, 0, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    /* Find out who is going to receive Bij */
    int s_dot = truncatedDivisionRemainer((processRow - processColumn + processStack*pc3), pc);
    int receiverRankForB = s_dot*dCol + processStack*dRow*dCol + processColumn;
    MPI_Sendrecv_replace(localB, localK * localN, MPI_DOUBLE, receiverRankForB, 1, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    double afterFirstShift = MPI_Wtime();
    #ifdef DEBUG
        printf("Rank: %d will send A%d%d to P%d%d%d and B%d%d to P%d%d%d\n", rank, processRow, processColumn, processRow, s, processStack, processRow, processColumn, s_dot, processColumn, processStack);
    #endif

    /* Assume that matrices have arrived */
    /* If your processStack is 0, meaning that you have the original C matrix, you need to add + beta*Cij to the sum. */
    double tempBeta = (processStack == 0) ? beta : 0.00000;
    double beforeFirstExec = MPI_Wtime();
    // double tempBeta = 0;
    cublasDgemm(cublasContext, charToCublasTransOp(TransA), charToCublasTransOp(TransB), 
        localM, localN, localK, &alpha, localA, llda, localB, lldb, &tempBeta, localC, lldc);
    double afterFirstExec = MPI_Wtime();
    
    /* If c = p^(1/3), the algorithm should end here since we basically run 3D SUMMA. If not, then we need to compute more tiles */
    int rotationCount = 1;
    s = (processColumn + rotationCount) % pc;
    s_dot = (processRow + rotationCount) % pc;
    if (rank == 0) {
        printf("First Stage - Shift %lf Exec %lf\n", afterFirstShift-beforeFirstShift, afterFirstExec-beforeFirstExec);
    }
    for (int i = rotationCount; i < pc3; i++) {
        double beforeNextShift = MPI_Wtime();
        /* Check if any of these are in local memory */
        receiverRankForA = processRow*dCol + processStack*dRow*dCol + s;
        MPI_Sendrecv_replace(localA, localM * localK, MPI_DOUBLE, receiverRankForA, 0, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        receiverRankForB = s_dot*dCol + processStack*dRow*dCol + processColumn;
        MPI_Sendrecv_replace(localB, localK * localN, MPI_DOUBLE, receiverRankForB, 1, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        double afterNextShift = MPI_Wtime();
        double tempBeta = 1.00;
        double beforeNextExec = MPI_Wtime();
        /* Calculate Cijk again - local matrices should be in GPU, else, you have to copy it before calling dgemm */
        cublasDgemm(cublasContext, charToCublasTransOp(TransA), charToCublasTransOp(TransB), 
            localM, localN, localK, &alpha, localA, llda, localB, lldb, &tempBeta, localC, lldc);
        double afterNextExec = MPI_Wtime();
        #ifdef DEBUG
            printf("Rank: %d will send A%d%d to P%d%d%d and B%d%d to P%d%d%d\n", rank, processRow, processColumn, processRow, s, processStack, processRow, processColumn, s_dot, processColumn, processStack);
        #endif  
        if (rank == 0) {
            printf("Next Stage - Shift %lf Exec %lf\n", afterNextShift-beforeNextShift, afterNextExec-beforeNextExec);
        }

        rotationCount++;
        s = (processColumn + rotationCount) % pc;
        s_dot = (processRow + rotationCount) % pc;
    }

    /* Reduce results */
    reduceResult();

    return;
}

/*  
    Gather C tiles to each process of common stack rank 
    Contribute Cijk with a sum-reduction to Pij0 
    In the end, everyone should have the final version of Cij on their local memory
    Call re-order to actually get result 
*/
void Summa25Decomposer::reduceResult()
{   
    /* Call Reduce */
    MPI_Reduce(MPI_IN_PLACE,
        localC,
        localM*localN,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        commonStackCommunicator
    );

    return;
}

/* Re-order matrix to original rank */
void Summa25Decomposer::reorderMatrix(int gatherRank, double* C)
{
    if (processStack != 0)
        return;

    int sendCount = 1, receiveCount = 0;
    if (rank == gatherRank) {
        sendCount = 0;
        receiveCount = (dRow*dCol - 1);
    }

    MPI_Request* sendRequests;
    if (sendCount > 0) {
        sendRequests = new MPI_Request[sendCount];
    }

    MPI_Request* receiveRequests;
    if (receiveCount > 0) {
        receiveRequests = new MPI_Request[receiveCount];
    }

    int sendIndex = 0, receiveIndex = 0;

    for (int i = 0; i < dRow; i++) {
        for (int j = 0; j < dCol; j++) {
            int senderRank = i*dCol + j;

            if (rank == gatherRank) {
                if (rank == senderRank) {
                    copyBlock(localM, localN, localC, &C[localM*(j*M + i)], localM, M);
                }
                else {
                    receiveBlock(localM ,localN, &C[localM*(j*M + i)], M, senderRank, 0, &receiveRequests[receiveIndex++]);
                }
            }

            if ((rank == senderRank) && (rank != gatherRank)) {
                transferBlock(localM, localN, localC, localM, gatherRank, 0, &sendRequests[sendIndex++]);
            }
        }
    }
    
    if (receiveIndex > 0) {
        MPI_Waitall(receiveIndex, receiveRequests, MPI_STATUSES_IGNORE);
        delete[] receiveRequests;
    }

    if (sendIndex > 0) {
        MPI_Waitall(sendIndex, sendRequests, MPI_STATUSES_IGNORE);
        delete[] sendRequests;
    }

    return;
}

Summa25Decomposer::~Summa25Decomposer()
{
    /* Delete cuBLAS context */
    CUBLAS_CHECK(cublasDestroy(cublasContext));
}

void fullOffloadSumma25Dgemm(char TransA, char TransB, long long M, long long N, long long K, 
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
            std::string filename = "DGEMM_execution_logs-FullOffload_GEMM-" + machineName + "-2.5DSUMMA-cuBLAS.csv";
            std::string header = "Algo,M,N,K,TotalNodes,TotalGPUs,dRow,dCol,StackSize,DecompositionTime,ExecutionTime,GatherTime,GFlops,DataLocation";
            logfile = createLogCsv(filename, header);
        }
    }

    /* Decompose Original Matrices */
    double beforeDecomp = MPI_Wtime();
    Summa25Decomposer decomposer(M, N, K, dRow, dCol, c);
    decomposer.decompose2D(0, A, B, C);
    double afterDecomp = MPI_Wtime();

    /* Warmup runs */
    for (int i = 0; i < 10; i++) {
        /* Warmup...*/
    }

    /* Actual runs */
    for (int i = 0; i < numberOfRuns; i++) {
        double beforeExecution = MPI_Wtime();
        decomposer.multiply(TransA, TransB, A, B, C, alpha, beta);
        double afterExecution = MPI_Wtime();

        double beforeGather = MPI_Wtime();
        if (gatherResults) {
            decomposer.reorderMatrix(0, C);
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
                                "2.5D SUMMA cuBLAS", M, N, K, numberOfNodes, size, dRow, dCol, c, decompTime, executionTime, gatherTime, gflops, "devices");
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


void preDistributedSumma25Dgemm(char TransA, char TransB, long long M, long long N, long long K, 
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
            std::string filename = "DGEMM_execution_logs-PreDistributed_GEMM-" + machineName + "-2.5DSUMMA-cuBLAS.csv";
            std::string header = "Algo,M,N,K,TotalNodes,TotalGPUs,dRow,dCol,StackSize,DecompositionTime,ExecutionTime,GatherTime,GFlops,DataLocation";
            logfile = createLogCsv(filename, header);
        }
    }

    /* Create Decomposer */
    double beforeDecomp = MPI_Wtime();
    Summa25Decomposer decomposer(M, N, K, dRow, dCol, c);
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
        decomposer.multiply(TransA, TransB, A, B, C, alpha, beta);
        double afterExecution = MPI_Wtime();

        double beforeGather = MPI_Wtime();
        if (gatherResults) {
            decomposer.reorderMatrix(0, C);
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
                                "2.5D SUMMA cuBLAS", M, N, K, numberOfNodes, size, dRow, dCol, c, decompTime, executionTime, gatherTime, gflops, "devices");
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
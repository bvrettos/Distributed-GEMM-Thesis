#include "2DBlockSequentialDecomposition.hpp"
// #define DEBUG

BlockSequentialDecomposer::BlockSequentialDecomposer(int M, int N, int K, MPI_Comm communicator, bool colMajor) : M(M), N(N), K(K), GEMM_Communicator(communicator), colMajor(colMajor)
{
    MPI_CHECK(MPI_Comm_rank(GEMM_Communicator, &rank));
    MPI_CHECK(MPI_Comm_size(GEMM_Communicator, &size));
    numberOfDevices = size;
    calculateVirtualDeviceGrid();

    localM = M/dRow;
    localN = N/dCol;
    localK = K;

    /* No need for process communication - do not create a process grid, just calculate it */
    processRow = rank/dCol;
    processColumn = rank%dCol;

    if ((processRow + 1) == dRow) {
        localM += M%dRow;
    }
    if ((processColumn + 1) == dCol) {
        localN += N%dCol;
    }
}

void BlockSequentialDecomposer::calculateVirtualDeviceGrid()
{
    int Px = std::sqrt(numberOfDevices);
    int Py = Px;

    /* If less than 4 devices */
    if (Px == 0) {
        Py = numberOfDevices;
        Px = 1;
    }
    
    /* If more than 4 devices, find the most square decomposition */
    int counter;
    for (counter = Px; counter > 0; --counter) 
        if (numberOfDevices % counter == 0) break;
    
    if (counter==0) {
        Px = numberOfDevices;
        Py = 1;
    }
    else {
        Px = counter;
        Py = numberOfDevices/counter;
    }

    dRow = Py;
    dCol = Px;
}

void BlockSequentialDecomposer::deliverMatrix(double* globalA, double* globalB, double* globalC, double** localA, double** localB, double** localC)
{
    /* Sender is rank = 0. Number of Tiles are 3 per rank, remove rank==0 tiles because they are transfered locally. So, number of tiles are (size-1)*3 */
    int sendCount = 0, receiveCount = 3;
    if (rank == 0) {
        sendCount = (size-1)*3;
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

    int sendIndex = 0;
    int receiveIndex = 0;    

    /* Matrix belongs to rank 0, total blocks are dRow * dCol. */
    long long tileK = K;
    for (int i = 0; i < dRow; i++) {
        long long tileM = M/dRow;
        if (i+1 == dRow)
            tileM += M%dRow;

        for (int j = 0; j < dCol; j++) {
            int receiverRank = i*dCol + j;
            long long tileN =  N/dCol;
            if (j+1 == dCol)
                tileN += N%dCol;
        
            /* Row-Major Offsets */
            int aOffset = (M/dRow)*K*i;
            int bOffset = (N/dCol)*j;
            int cOffset = (M/dRow)*N*i + (N/dCol)*j;
            long long llda, lldb, lldc;
            long long lda, ldb, ldc;
            lda = K;
            ldb = N;
            ldc = N;
            llda = tileK;
            lldb = tileN;
            lldc = tileN;

            if (colMajor) {
                aOffset = (M/dRow)*i;
                bOffset = (N/dCol)*K*j;
                cOffset = (N/dCol)*M*j + (M/dRow)*i;
                lda = M;
                ldb = K;
                ldc = M;
                llda = tileM;
                lldb = tileK;
                lldc = tileM;
            }

            #ifdef DEBUG
                if (rank == 0) {
                    printf("Rank %d needs to receive tiles of dimensions A(%d, %d), B(%d, %d), C(%d, %d)\n", receiverRank, tileM, tileK, tileK, tileN, tileM, tileN);
                    printf("Current Strides: A[%d], B[%d], C[%d]\n", aOffset, bOffset, cOffset);
                }
            #endif

            /* Receive Blocks */
            if ((rank == receiverRank) && (rank != 0)) {
                receiveBlock(tileM, tileK, *localA, llda, 0, 0, &receiveRequests[receiveIndex++], colMajor);
                receiveBlock(tileK, tileN, *localB, lldb, 0, 0, &receiveRequests[receiveIndex++], colMajor);
                receiveBlock(tileM, tileN, *localC, lldc, 0, 0, &receiveRequests[receiveIndex++], colMajor);
            }
            
            /* Let's say that memory has already been allocated */
            if (rank == 0) {
                if (rank == receiverRank) {
                    /* Copy blocks to local memory instead of sending blocks through MPI */
                    copyBlock(tileM, tileK, globalA, *localA, lda, llda, colMajor);
                    copyBlock(tileK, tileN, globalB, *localB, ldb, lldb, colMajor);
                    copyBlock(tileM, tileN, globalC, *localC, ldc, lldc, colMajor);
                }
                else {
                    /* Send Tiles */
                    transferBlock(tileM, tileK, &globalA[aOffset], lda, receiverRank, 0, &sendRequests[sendIndex++], colMajor);
                    transferBlock(tileK, tileN, &globalB[bOffset], ldb, receiverRank, 0, &sendRequests[sendIndex++], colMajor);
                    transferBlock(tileM, tileN, &globalC[cOffset], ldc, receiverRank, 0, &sendRequests[sendIndex++], colMajor);
                }
            }
        }
    }

    for (int i = 0; i < receiveCount; i++) {
        int idx;
        MPI_Waitany(receiveIndex, receiveRequests, &idx, MPI_STATUS_IGNORE);
    }
    
    if (receiveCount > 0) {   
        delete[] receiveRequests;
    }

    /* After sending all requests, then wait */
    if (sendCount > 0) {
        MPI_Waitall(sendIndex, sendRequests, MPI_STATUSES_IGNORE);
        delete[] sendRequests;
    }

    // printMatrix(*localA, localM, localK, rank);
}

void BlockSequentialDecomposer::gatherResult(int gatherRank, double* C, double* localC)
{
    int sendCount = 1, receiveCount = 0;
    if (rank == gatherRank) {
        sendCount = 0;
        receiveCount = size-1;
    }

    MPI_Request* sendRequests;
    if (sendCount > 0) 
        sendRequests = new MPI_Request[sendCount];
    
    MPI_Request* receiveRequests;
    if (receiveCount > 0)
        receiveRequests = new MPI_Request[receiveCount];

    int sendIndex = 0;
    int receiveIndex = 0;

    long long tileK = K;

    for (int i = 0; i < dRow; i++) {
        long long tileM = M/dRow;
        if (i+1 == dRow)
            tileM += M%dRow;
        
        for (int j = 0; j < dCol; j++) {
            int senderRank = i*dCol + j;

            long long tileN = N/dCol;
            if (j+1 == dCol)
                tileN += N%dCol;

            /* Row-Major Offsets */
            int cOffset = (M/dRow)*N*i + (N/dCol)*j;
            int senderCOffset = (M/dRow)*N*processRow + (N/dCol)*processColumn;
            long long lldc;
            long long ldc;

            ldc = N;
            lldc = tileN;

            if (colMajor) {
                cOffset = (N/dCol)*M*j + (M/dRow)*i;
                senderCOffset = (N/dCol)*M*processColumn + (M/dRow)*processRow;
                ldc = M;
                lldc = tileM;
            }

            #ifdef DEBUG
                if (rank == 0) {
                    printf("Will Gather C(%d, %d)\n", tileM, tileN);
                    printf("Current Stride: C[%d]\n", cOffset);
                }
            #endif

            if ((rank == senderRank) && (rank != gatherRank)) {
                /* Send Blocks */
                transferBlock(tileM, tileN, localC, lldc, gatherRank, 0, &sendRequests[sendIndex++], colMajor);
            }

            if (rank == gatherRank) {
                if (rank == senderRank) {
                    /* Instead of sending block to yourself, copy from local memory */
                    copyBlock(tileM, tileN, localC, &C[senderCOffset], lldc, ldc, colMajor);
                }
                else {
                    /* Receive Blocks from others */
                    receiveBlock(tileM, tileN, &C[cOffset], ldc, senderRank, 0, &receiveRequests[receiveIndex++], colMajor);
                }
            }
        }
    }

    /* Waitall for receivers/senders */
    if (receiveCount > 0) {
        for (int i = 0; i < receiveIndex; i++) {
            int idx;
            MPI_Waitany(receiveIndex, receiveRequests, &idx, MPI_STATUS_IGNORE);
        }
        delete[] receiveRequests;
    }

    if (sendCount > 0) {
        MPI_Waitall(sendIndex, sendRequests, MPI_STATUSES_IGNORE);
        delete[] sendRequests;
    }

    return;
}

BlockSequentialDecomposer::~BlockSequentialDecomposer()
{

}
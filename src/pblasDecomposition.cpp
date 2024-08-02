#include "pblasDecomposition.hpp"

int numroc(int n, int nb, int iproc, int isrproc, int nprocs)
{
    int extraBlocks, myDistance, numBlocks;
    int numroc = 0;

    myDistance = (nprocs + iproc - isrproc) % nprocs;

    numBlocks = n / nb;
    numroc = (numBlocks/nprocs) * nb;

    extraBlocks = numBlocks % nprocs;

    if (myDistance < extraBlocks) {
        numroc += nb;
    }
    else if (myDistance == extraBlocks) {
        numroc += n % nb;
    }

    return numroc;
}

pblasDecomposer::pblasDecomposer(int M, int N, int Mb, int Nb, MPI_Comm communicator) : M(M), N(N), Mb(Mb), Nb(Nb), communicator(communicator)
{
    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &size);

    /* Process Grid (Row-Major)*/
    calculateProcessGrid(&dRow, &dCol, size);
    int rootRow = 0;
    int rootCol = 0;

    procRow = rank/dCol;
    procColumn = rank%dCol;

    /* Block Grid */
    gridRows = M/Mb;
    if (M%Mb)
        gridRows++; // Extra blocks
    gridColumns = N/Nb;
    if (N%Nb)
        gridColumns++; // Extra blocks

    #ifdef DEBUG
        if (rank == 0)
            printf("Grid: %d x %d blocks\n", gridRows, gridColumns);
    #endif

    /* Local Matrix Dimensions */
    localRows = numroc(M, Mb, procRow, rootCol, dRow);
    localColumns = numroc(N, Nb, procColumn, rootRow, dCol);

    myBlocks=0;

    for (int i = 0; i < gridRows; i++) {
        if (i%dRow == procRow) {
            for (int j = 0; j < gridColumns; j++) {
                if (j%dCol == procColumn) {
                    myBlocks++;
                }
            }
        }
    }

    #ifdef DEBUG
        printf("Rank: %d has %d blocks in %lldx%lld Matrix with %dx%d tiling\n", rank, myBlocks, M, N, Mb, Nb);
        printf("Rank: %d, procRow %d procCol %d\n");
    #endif
}

void pblasDecomposer::scatterMatrix(int senderRank, double* globalMatrix, double* localMatrix)
{
    int receiverProcessRow = 0, receiverProcessColumn = 0;
    int rowOffset = 0, columnOffset = 0;
    int sendIndex=0, receiveIndex=0;
    
    int sendCount = 0;
    int receiveCount = myBlocks;
    if (rank == senderRank) {
        sendCount = gridRows*gridColumns - myBlocks;
        receiveCount = 0;
    }

    // printf("Rank %d has %d send and %d receive requests\n", rank, sendCount, receiveCount);

    MPI_Request* sendRequests;
    MPI_Request* receiveRequests;
    if (sendCount > 0)
        sendRequests = new MPI_Request[sendCount];
    if (receiveCount > 0)
        receiveRequests = new MPI_Request[receiveCount];

    for (int row = 0; row < M; row += Mb, receiverProcessRow = (receiverProcessRow+1) % dRow) {
        receiverProcessColumn = 0;

        int rowsToSend = Mb;
        /* Is this last row block?*/
        if (M - row < Mb)
            rowsToSend = M - row;
            
        for (int col = 0; col < N; col += Nb, receiverProcessColumn = (receiverProcessColumn+1) % dCol) {
            int receiverRank = receiverProcessRow*dCol + receiverProcessColumn;
            int columnsToSend = Nb;
            /* Is this last column block? */
            if (N - col < Nb)
                columnsToSend = N - col;

            /* Send block*/
            if (rank == senderRank) {
                if (rank == receiverRank)
                    copyBlock(rowsToSend, columnsToSend, &globalMatrix[col*M + row], &localMatrix[columnOffset*localRows + rowOffset], M, localRows); /* If you are the owner of the global matrix, copy block locally */
                else
                    transferBlock(rowsToSend, columnsToSend, &globalMatrix[col*M + row], M, receiverRank, 0, &sendRequests[sendIndex++]);
            }

            /* Receive block*/
            if ((rank == receiverRank) && (rank != senderRank)) {
                receiveBlock(rowsToSend, columnsToSend, &localMatrix[columnOffset*localRows + rowOffset], localRows, senderRank, 0, &receiveRequests[receiveIndex++]);
            }

            /* Update offset */
            if (rank == receiverRank)
                columnOffset = (columnOffset + columnsToSend) % localColumns;
        }

        /* If you just got sent a block, then change your row offset */
        if (procRow == receiverProcessRow) 
            rowOffset = (rowOffset + rowsToSend) % localRows;
    }

    /* Trading phase over, wait for tiles */
    for (int i = 0; i < receiveCount; i++) {
        int idx;
        MPI_Waitany(receiveIndex, receiveRequests, &idx, MPI_STATUS_IGNORE);
    }

    if (receiveCount > 0)
        delete[] receiveRequests;
    
    if (sendCount > 0) {
        MPI_Waitall(sendIndex, sendRequests, MPI_STATUSES_IGNORE);
        delete[] sendRequests;
    }

    return;
}

void pblasDecomposer::gatherMatrix(int receiverRank, double* globalMatrix, double* localMatrix)
{
    /* Gather matrix */
    int senderProcessRow = 0, senderProcessColumn = 0;
    int rowOffset=0, columnOffset=0;

    int sendIndex = 0, receiveIndex = 0;
    
    int sendCount = myBlocks; // Each process sends their blocks
    int receiveCount = 0;
    if (rank == receiverRank) {
        sendCount = 0;
        receiveCount = gridRows*gridColumns - myBlocks; // Receiver rank in total gets all of the blocks minus their own.
    }

    MPI_Request* sendRequests;
    MPI_Request* receiveRequests;
    if (sendCount > 0)
        sendRequests = new MPI_Request[sendCount];
    if (receiveCount > 0)
        receiveRequests = new MPI_Request[receiveCount];

    for (int row = 0; row < M; row += Mb, senderProcessRow=(senderProcessRow+1) % dRow) {
        senderProcessColumn = 0;

        // Is this the last row block?
        int rowsToReceive = Mb;
        if (M - row < Mb)
            rowsToReceive = M - row;
 
        for (int col = 0; col < N; col += Nb, senderProcessColumn = (senderProcessColumn+1) % dCol) {
            // Number of cols to be sent
            // Is this the last col block?
            int columnsToReceive = Nb;
            if (N - col < Nb)
                columnsToReceive = N - col;

            int senderRank = senderProcessRow*dCol + senderProcessColumn;
 
            if ((rank == senderRank) && (rank != receiverRank)) {
                transferBlock(rowsToReceive, columnsToReceive, &localMatrix[columnOffset*localRows + rowOffset], localRows, receiverRank, 0, &sendRequests[sendIndex++]);
            }
 
            if (rank == receiverRank) {
                if (rank == senderRank)  /* Copy the matrix locally*/
                    copyBlock(rowsToReceive, columnsToReceive, &localMatrix[columnOffset* localRows + rowOffset], &globalMatrix[col*M + row], localRows, M);
                else 
                    receiveBlock(rowsToReceive, columnsToReceive, &globalMatrix[col*M + row], M, senderRank, 0, &receiveRequests[receiveIndex++]);
            }

            if (rank == senderRank) 
                columnOffset = (columnOffset + columnsToReceive) % localColumns; 
        }
 
        if (procRow == senderProcessRow) 
            rowOffset = (rowOffset + rowsToReceive) % localRows;
    }

    /* Trading phase over, wait for tiles */
    for (int i = 0; i < receiveCount; i++) {
        int idx;
        MPI_Waitany(receiveIndex, receiveRequests, &idx, MPI_STATUS_IGNORE);
    }

    if (receiveCount > 0)
        delete[] receiveRequests;
    
    if (sendCount > 0) {
        MPI_Waitall(sendIndex, sendRequests, MPI_STATUSES_IGNORE);
        delete[] sendRequests;
    }
}

pblasDecomposer::~pblasDecomposer()
{

}
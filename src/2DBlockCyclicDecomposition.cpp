#include <2DBlockCyclicDecomposition.hpp>

typedef enum {
    2D_BLOCK,
    2D_BLOCK_CYCLIC
} DecompositionType;

/* Tiles:
    0(0,0)(0) 1(0,1)(1) 2(0,2)(2) 3(0,3)(0)
    4(1,0)(3) 5(1,1)(4) 6(1,2)(5) 7(1,3)(3)
    8(2,0) 9(2,1) 10(2,2) 11(2,3)
    12(3,0) 13(3,1) 14(3,2) 15(3,3)
*/

/* Procs:
    0(0,0) 1(0,1) 2(0,2)
    3(1,0) 4(1,1) 5(1,2)
*/

bool BlockCyclicMatrixDecomposer::isTileMineScalapack(int tileRow, int tileColumn)
{   
    return ((tileRow%procRow == 0) && (tileColumn%procColumn == 0));
}

BlockCyclicMatrixDecomposer::BlockCyclicMatrixDecomposer(int rows, int columns, int blockRows, int blockColumns, MPI_Comm communicator, DecompositionType decompositionType) : M(rows), N(columns), blockRows(blockRows), blockColumns(blockColumns), communicator(communicator), decompositionType(decompositionType)
{
    rowDiv = rows/blockRows;
    rowMod = rows%blockRows;
    colDiv = columns / blockColumns;
    colMod = columns % blockColumns;

    gridRows = (rows + blockRows - 1) / blockRows;
    gridColumns = (columns + blockColumns - 1) / blockColumns;

    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &size);

    calculateVirtualDeviceGrid();

    /* 2D Process Grid */
    procRow = rank / dCol;
    procCol = rank % dCol;

    switch(decompositionType) {
        case 2D_BLOCK:
            
            break;

        case 2D_BLOCK_CYCLIC:

            break;
    }
}

void BlockCyclicMatrixDecomposer::calculateVirtualDeviceGrid()
{
    int Px = std::sqrt(size);
    int Py = Px;

    /* If less than 4 devices */
    if (Px == 0) {
        Py = size;
        Px = 1;
    }
    
    /* If more than 4 devices, find the most square decomposition */
    int counter;
    for (counter = Px; counter > 0; --counter) 
        if (size % counter == 0) break;
    
    if (counter==0) {
        Px = size;
        Py = 1;
    }
    else {
        Px = counter;
        Py = size/counter;
    }

    dRow = Py;
    dCol = Px;
}

void BlockCyclicMatrixDecomposer::decomposeScalapack()
{
    /* Create 3 pblas Decomposers */
    pblasDecomposer MatrixA(M, K, Mb, Nb, MPI_COMM_WORLD), MatrixB(K, N, Mb, Nb, MPI_COMM_WORLD), MatrixC(M, N, Mb, Nb, MPI_COMM_WORLD);

    /* Allocate space in host/device */
    double *localA, *localB, *localC;

    /* Scatter matrices */
    MatrixA.scatterMatrix(0, A, localA);
    MatrixB.scatterMatrix(0, B, localB);
    MatrixC.scatterMatrix(0, C, localC);

    /* Find out which tiles belong to who */

    return;
}


/* Steal this from 2.5D decompose2D method */
void BlockCyclicMatrixDecomposer::decomposeCannon(int senderRank, double* A, double* B, double* C)
{
    /* 
        Original SUMMA decomposition should only happen on the grid with k = 0. Don't know if 
        straight up returning is the best method
    */
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

void BlockCyclicMatrixDecomposer::multiplySummaCyclic()
{
    return;
}

void BlockCyclicMatrixDecomposer::multiplySumma()
{
    return;
}

void BlockCyclicMatrixDecomposer::multiplyCannon()
{
    /* Same as 2.5D summa with processStack=1 */
    return;
}

BlockCyclicMatrixDecomposer::~BlockCyclicMatrixDecomposer()
{

}
#include <DistributedMatrix.hpp>
#include <vector>

template <typename scalar_t>
void DistributedMatrix<scalar_t>::distribute(int rootRank)
{
    /* No need for numroc, memory is initialized per Tile */

    int receiverProcessRow = 0, receiverProcessColumn = 0;
    int gridRowIndex = 0, gridColumnIndex = 0;
    int sendIndex = 0, receiveIndex = 0;

    /* No need to keep track of max sends/receives */
    std::vector<MPI_Request> sendRequests;
    std::vector<MPI_Request> receiveRequests;

    for (int i = 0; i < rows; i += blockRows) { // Iterate over rows of global matrix
        receiverProcessColumn = 0;
        int rowsToSend = blockRows;
        
        /* Is this the last row block (with modulo?)*/
        if (rows - i < blockRows)
            rowsToSend = rows - i;

        for (int j = 0; j < columns; j += blockColumns) {
            int receiverRank = receiverProcessRow*dCol + receiverProcessColumn;
            int columnsToSend = blockColumns;
            long offset = 0;
            /* Is this the last column block? (with modulo) */
            if (columns - j < blockColumns)
                columnsToSend = columns - j;
            

            /* Calculate offset of Global Matrix - TODO: There should be an easier way (without using swap :()*/
            if (layout == MatrixLayout::ColumnMajor)
                offset = j*ld + i;
            else if (layout == MatrixLayout::RowMajor)
                offset = i*ld + j;

            /* Everyone should create the tile in the map, but only some will have access to it's data */
            tileMap[gridRowIndex][gridColumnIndex] = Tile<scalar_t>(rowsToSend, columnsToSend, this->location, layout); // This does not allocate memory
            Tile<scalar_t>& currentTile;
            currentTile = tileMap[gridRowIndex][gridColumnIndex];

            /* Actually send block */
            if (rank == rootRank) {
                if (rank == receiverRank) {
                    currentTile.allocateMemory();
                    memcpy2D(&this->matrixData[offset], this->ld, currentTile.getDataPointer(), currentTile.getStride(), 
                        rowsToSend, columnsToSend, sizeof(scalar_t), layout);
                }
                    
                else {
                    MPI_Request request;
                    transferBlock(rowsToSend, columnsToSend, &this->matrixData[offset], this->ld, receiverRank, 0, &request);
                    sendRequests.push_back(request);
                    sendIndex++;
                }
            }

            if ((rank == receiverRank) && (rank != rootRank)) {
                // Receive Block - Allocate memory on Tile before receiving
                MPI_Request request;
                currentTile.allocateMemory();
                currentTile.irecv(rootRank, mpiCommunicator, layout, 0, &request);
                receiveRequests.push_back(request);
                receiveIndex++;
            }

            /* Update Tile Grid column index and next process to receive*/
            gridColumnIndex++;
            receiverProcessColumn = (receiverProcessColumn+1) % dCol;
        }

        /* Update Tile Grid row index and next process to receive */
        gridRowIndex++;
        gridColumnIndex = 0;
        receiverProcessRow = (receiverProcessRow+1) % dRow;
    }

    /* Wait for tiles to be received */
    if (receiveIndex > 0) {
        MPI_Waitall(receiveIndex, receiveRequests.data(), MPI_STATUSES_IGNORE);
        receiveRequests.clear();
    }

    if (sendIndex > 0) {
        MPI_Waitall(sendIndex, sendRequests.data(), MPI_STATUSES_IGNORE);
        sendRequests.clear();
    }
    /* Let vectors go out of scope - they will delete themselves*/

    return;
}

template <typename scalar_t>
void DistributedMatrix<scalar_t>::gather(scalar_t* A, int64_t lda, int rootRank)
{
    /* Gather matrix */
    int senderProcessRow = 0, senderProcessColumn = 0;
    int gridRowIndex = 0, gridColumnIndex = 0;
    int sendIndex = 0, receiveIndex = 0;

    /* No need to keep track of max sends/receives */
    std::vector<MPI_Request> sendRequests;
    std::vector<MPI_Request> receiveRequests;
    bool colMajor = true;
    if (layout == MatrixLayout::RowMajor)
        colMajor = false;

    for (int i = 0; i < rows; i += blockRows) {
        senderProcessColumn = 0;

        // Is this the last row block?
        int rowsToReceive = blockRows;
        if (rows - i < blockRows)
            rowsToReceive = rows - i;
 
        for (int j = 0; j < columns; j += blockColumns) {
            // Number of cols to be sent
            // Is this the last col block?
            int columnsToReceive = blockColumns;
            if (columns - j < blockColumns)
                columnsToReceive = columns - j;

            /* Calculate offset of Global Matrix - TODO: There should be an easier way (without using swap :()*/
            long offset = 0;
            if (layout == MatrixLayout::ColumnMajor)
                offset = j*ld + i;
            else if (layout == MatrixLayout::RowMajor)
                offset = i*ld + j;

            Tile<scalar_t>& currentTile;
            currentTile = tileMap[gridRowIndex][gridColumnIndex];

            int senderRank = senderProcessRow*dCol + senderProcessColumn;
 
            if ((rank == senderRank) && (rank != rootRank)) {
                /* If not the root rank, send your tile to master process */
                MPI_Request request;
                currentTile.isend(rootRank, mpiCommunicator, 0, &request);
                sendRequests.push_back(request);
                sendIndex++;
            }
 
            if (rank == rootRank) {
                if (rank == senderRank) {
                    /* Copy the matrix locally*/
                    memcpy2D(currentTile.getDataPointer(), currentTile.getStride(), &A[offset], lda,
                        rowsToReceive, columnsToReceive, sizeof(scalar_t), colMajor);
                }  
                else {
                    MPI_Request request;
                    receiveBlock(rowsToReceive, columnsToReceive,  &A[offset], lda, 
                        senderRank, 0, &request, colMajor);
                    receiveRequests.push_back(request);
                    receiveIndex++;
                }
            }

            gridColumnIndex++;
            senderProcessColumn = (senderProcessColumn+1) % dCol;
        }

        gridColumnIndex = 0;
        gridRowIndex++;
        senderProcessRow=(senderProcessRow+1) % dRow;
    }

    /* Wait for tiles to be received */
    if (receiveIndex > 0) {
        MPI_Waitall(receiveIndex, receiveRequests.data(), MPI_STATUSES_IGNORE);
        receiveRequests.clear();
    }

    if (sendIndex > 0) {
        MPI_Waitall(sendIndex, sendRequests.data(), MPI_STATUSES_IGNORE);
        sendRequests.clear();
    }
    
    /* Let vectors go out of scope - they will delete themselves*/

    return;
}
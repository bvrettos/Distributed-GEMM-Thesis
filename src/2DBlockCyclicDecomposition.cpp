#include <2DBlockCyclicDecomposition.hpp>

BlockCyclicMatrixDecomposer::BlockCyclicMatrixDecomposer(int rows, int columns, int blockRows, int blockColumns, MPI_Comm communicator) : M(rows), N(columns), blockRows(blockRows), blockColumns(blockColumns), communicator(communicator)
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
    procRow = rank / dCol;
    procCol = rank % dCol;
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

BlockCyclicMatrixDecomposer::~BlockCyclicMatrixDecomposer()
{

}
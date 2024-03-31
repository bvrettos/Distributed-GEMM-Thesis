#include <2DBlockCyclicDecomposition.hpp>

BlockCyclicMatrixDecomposer::BlockCyclicMatrixDecomposer(int rows, int columns, int blockRows, int blockColumns, MPI_Comm communicator) : M(rows), N(columns), blockRows(blockRows), blockColumns(blockColumns)
{
    this->communicator = communicator;

    rowDiv = rows/blockRows;
    rowMod = rows%blockRows;

    colDiv = columns / blockColumns;
    colMod = columns % blockColumns;

    gridRows = (rows + blockRows - 1) / blockRows;
    gridColumns = (columns + blockColumns - 1) / blockColumns;

    hasHorizontal = false;
    hasVertical = false;
    hasSmall = false;

    /* Create block types */
    if (rowMod) {
        /* Create horizontal type */
        hasHorizontal = true;
    }

    if (colMod) {
        /* Create Vertical type */
        hasVertical = true;
    }

    if (colMod && rowMod) {
        /* Create smol type */
        hasSmall = true;
    }

    MPI_Comm_rank(communicator, &rank);
    MPI_Comm_size(communicator, &size);

    calculateVirtualDeviceGrid();
    procRow = rank / dCol;
    procCol = rank % dCol;

    allocateLocalMatrix();
}

void BlockCyclicMatrixDecomposer::allocateLocalMatrix()
{
    /* Find out how many tiles (and what type of tiles) each process gets */
    int totalTiles = gridRows*gridColumns;

    if (size > totalTiles) {
        std::cout << "Too many procs, remove some" << std::endl;
        exit(1);
    }

    int tilesPerRank = totalTiles/size;
    int extraTiles = totalTiles%size;
    int rankTiles[size];

    for (int i = 0; i < size; i++) {
        rankTiles[i] = tilesPerRank;
        if (extraTiles > 0) {
            rankTiles[i]++;
            extraTiles--;
        }
    }

    int myTiles = rankTiles[rank];
    int myTilesArray[myTiles];



    /* Full retard mode but ok :( */
    printf("Rank: %d has %d tiles\n", rank, myTiles);
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
#include <DistributedMatrix.hpp>

template <typename scalar_t>
DistributedMatrix<scalar_t>::DistributedMatrix() :
    rows(0),
    columns(0),
    blockRows(0),
    blockColumns(0),
    tileGridRows(0),
    tileGridColumns(0),
    mpiCommunicator(nullptr),
    distribution(DistributionStrategy::BlockCyclic),
    layout(MatrixLayout::ColumnMajor),
    memoryLocation(MemoryLocation::Host),
    tileMap(nullptr),
    dStack(1)
{
    /* Get MPI Info */
    MPI_Comm_rank(mpiCommunicator, &rank);
    MPI_Comm_size(mpiCommunicator, &size);
}

template <typename scalar_t>
DistributedMatrix<scalar_t>::DistributedMatrix(long long rows, long long columns, int64_t ld, MemoryLocation location, MPI_Comm communicator,
    MatrixLayout layout, DistributionStrategy distribution) :
    rows(rows),
    columns(columns),
    ld(ld),
    memoryLocation(location),
    distribution(distribution),
    mpiCommunicator(communicator),
    layout(layout),
    dStack(1)
{
    /* Depending on DistributionStrategy, calculate tile map */
    MPI_Comm_rank(mpiCommunicator, &rank);
    MPI_Comm_size(mpiCommunicator, &size);

    /* Use 2D Block Cyclic algo to distribute matrix to location */

    /* Memory Allocation happens tile-wise */

}

template <typename scalar_t>
DistributedMatrix<scalar_t>::DistributedMatrix(long long rows, long long columns, int64_t ld, int blockRows, int blockColumns, MemoryLocation location, MPI_Comm communicator, MatrixLayout layout) :
    rows(rows),
    columns(columns),
    blockRows(blockRows),
    ld(ld),
    blockColumns(blockColumns),
    memoryLocation(location),
    distribution(distribution),
    mpiCommunicator(communicator),
    layout(layout),
    dStack(1)
{
    /* Get MPI Info - Calculate Process Grid */
    MPI_Comm_rank(mpiCommunicator, &rank);
    MPI_Comm_size(mpiCommunicator, &size);
    calculate2DProcessGrid();
    this->deviceId = rank;

    /* BlockRows and Block Columns selected, so Distribution can only be Block Cyclic . Calculate Tile Map */
    initializeTileMap();
}

template <typename scalar_t>
DistributedMatrix<scalar_t>::DistributedMatrix(long long rows, long long columns, int64_t ld, int blockRows, int blockColumns, int dRow, int dCol, MemoryLocation location, MPI_Comm communicator, 
    MatrixLayout layout) :
    rows(rows),
    columns(columns),
    ld(ld),
    memoryLocation(location),
    mpiCommunicator(communicator),
    dRow(dRow),
    dCol(dCol),
    layout(layout),
    distribution(distribution),
    dStack(1)
{
    MPI_Comm_rank(mpiCommunicator, &rank);
    MPI_Comm_size(mpiCommunicator, &size);


    /* Do not calculate process grid, this is provided by user - Calculate Tile Map */
    initializeTileMap();
}


template <typename scalar_t>
void DistributedMatrix<scalar_t>::initializeTileMap()
{
    rowModulo = false;
    columnModulo = false;

    /* First, find the dimensions of the tile grid */
    tileGridRows = rows/blockRows;
    if (rows%blockRows) {
        tileGridRows++;
        rowModulo = true;
    }
        
    tileGridColumns = columns/blockColumns;
    if (columns%blockColumns) {
        tileGridColumns++;
        columnModulo = true;
    }
        
    /* Allocate a 2D array for the tiles */
    tileMap = new Tile<scalar_t>*[tileGridRows]; 
    for (int i = 0; i < tileGridRows; i++) {
        tileMap[i] = new Tile<scalar_t>[tileGridColumns];
    }

    #ifdef DEBUG
        if (rank == 0) printf("Tile Grid Dimensions: %d x %d", tileGridRows, tileGridColumns);
    #endif

    /* Initialize Tiles */
    for (int i = 0; i < tileGridRows; i++) {
        for (int j = 0; j < tileGridColumns; j++) {
            
            long long currentRows = (rowModulo && (i+1 == tileGridRows)) ? rows%blockRows : blockRows;
            long long currentColumns = (columnModulo && (j+1 == tileGridColumns)) ? columns%blockColumns : blockColumns;

            #ifdef DEBUG
                if (rank == 0) printf("CurrentRows: %lld, CurrentColumns %lld\n", currentRows, currentColumns);
            #endif

            tileMap[i][j] = Tile<scalar_t>(currentRows, currentColumns, this->memoryLocation, layout);
            
            if (this->tileIsMine(i ,j)) {
                tileMap[i][j].allocateMemory();
            }
        }
    }

    return;
}

template <typename scalar_t>
DistributedMatrix<scalar_t> generateRandomMatrix(int64_t rows, int64_t columns, int blockRows, int blockColumns, 
    MemoryLocation location, MPI_Comm communicator, MatrixLayout layout)
{
    int64_t ld;
    if (layout == MatrixLayout::ColumnMajor) {
        ld = rows;
    }
    else 
        ld = columns;

    DistributedMatrix<scalar_t> matrix(rows, columns, ld, blockRows, blockColumns, location, communicator, layout);

    /* Iterate through tileMap and generate random tile if tile is yours */
    for (int i = 0; i < matrix.gridRows(); i++) {
        for (int j = 0; j < matrix.gridColumns(); j++) {
            if (matrix.tileIsMine(i, j)) {
                matrix.getTile(i, j).generateRandomValues(matrix.getDeviceId());
            }
        }
    }
    return matrix;
}

template <typename scalar_t>
int64_t DistributedMatrix<scalar_t>::getRows() { return this->rows; }

template <typename scalar_t>
int64_t DistributedMatrix<scalar_t>::getColumns() { return this->columns; }

template <typename scalar_t>
int DistributedMatrix<scalar_t>::gridRows() { return this->tileGridRows; }

template <typename scalar_t>
int DistributedMatrix<scalar_t>::gridColumns() { return this->tileGridColumns; }

template <typename scalar_t>
int DistributedMatrix<scalar_t>::getDeviceId() { return this->deviceId; }

template <typename scalar_t>
Tile<scalar_t>& DistributedMatrix<scalar_t>::getTile(int tileRow, int tileColumn) { return tileMap[tileRow][tileColumn]; }

template <typename scalar_t>
bool DistributedMatrix<scalar_t>::tileIsMine(int tileRow, int tileColumn) 
{
    return ((tileRow%dRow == processRow) && (tileColumn%dCol == processColumn));
}

template <typename scalar_t>
void DistributedMatrix<scalar_t>::print()
{
    /* Prints Matrix Information */
    if (this->rank == 0) {
        printf("Matrix(%lld x %lld) - BlockingDims(%d x %d) - TileGrid(%d x %d) \n", rows, columns, blockRows, blockColumns, tileGridRows, tileGridColumns);
        printf("ProcessGridDimensions(%d x %d x %d)\n", dRow, dCol, dStack);
        printf("----------------------------------------------------\n");
        printf("ProcessInfo(Rank: %d, ProcessRow: %d, ProcessColumn: %d, ProcessStack: %d) \n", rank, processRow, processColumn, processStack);
    }
}

template class DistributedMatrix<double>;
template class DistributedMatrix<float>;
template DistributedMatrix<double> generateRandomMatrix(int64_t rows, int64_t columns, int blockRows, int blockColumns, MemoryLocation location, MPI_Comm communicator, MatrixLayout layout);
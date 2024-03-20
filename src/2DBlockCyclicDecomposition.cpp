#include "2DBlockCyclicDecomposition.hpp"

BlockCyclicMatrixDecomposer::BlockCyclicMatrixDecomposer(int rows, int columns, int blockRows, int blockColumns) : matrixRows(rows), matrixColumns(columns), blockRows(blockRows), blockColumns(blockColumns)
{
        gridRows = (rows + blockRows - 1) / blockRows;
        gridColumns = (columns + blockColumns - 1) / blockColumns;

        std::cout << "Grid Rows: " << gridRows << " Grid Columns: " << gridColumns << std::endl;
        tileMap = (Tile**) malloc(sizeof(Tile*) * gridRows);
        for (int i = 0; i < gridRows; i++) {
            tileMap[i] = (Tile*) malloc(sizeof(Tile) * gridColumns);
        }
}

int BlockCyclicMatrixDecomposer::size()
{
    return gridRows*gridColumns;
}

void BlockCyclicMatrixDecomposer::decomposeMatrix() 
{
    for (int i = 0; i < gridRows; i++) {
        for (int j = 0; j < gridColumns; j++) {
            Tile tile;
            tile.sRow = i*blockRows;
            tile.sCol = j*blockColumns;

            tile.eRow = std::min((i+1)*blockRows, matrixRows);
            tile.eCol = std::min((j+1)*blockColumns, matrixColumns);

            tile.rowId = i;
            tile.colId = j;

            tileMap[i][j] = tile;
        }
    }
    return;
}

void BlockCyclicMatrixDecomposer::printTileMap() 
{
    for (int i = 0; i < gridRows; i++) {
        for (int j = 0; j < gridColumns; j++) {
            printTile(&tileMap[i][j]);
        }
    }
    return;
}

BlockCyclicMatrixDecomposer::~BlockCyclicMatrixDecomposer() {
    for (int i = 0; i < gridRows; i++) {
        free(tileMap[i]);
    }
    free(tileMap);
}

GEMM_BlockCyclicDecomposer::GEMM_BlockCyclicDecomposer(int M, int N, int K, int numberOfDevices, int blockRows, int blockColumns) : M(M), N(N), K(K), numberOfDevices(numberOfDevices), blockRows(blockRows), blockColumns(blockColumns), 
            A_Decomp(M, K, blockRows, blockColumns), B_Decomp(K, N, blockRows, blockColumns), C_Decomp(M, N, blockRows, blockColumns)
{
    int cTiles = C_Decomp.size();
    cTilesPerDevice = cTiles / numberOfDevices;

    taskMap = (Task**) malloc(sizeof(Task*) * numberOfDevices);
    for (int i = 0; i < numberOfDevices; i++) {
        taskMap[i] = (Task*) malloc(sizeof(Task) * cTilesPerDevice);
    }

    calculateVirtualDeviceGrid();

    A_Decomp.decomposeMatrix();
    B_Decomp.decomposeMatrix();
    C_Decomp.decomposeMatrix();
}

void GEMM_BlockCyclicDecomposer::calculateVirtualDeviceGrid()
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

#ifdef DEBUG
    printf("dRow: %d - dCol: %d\n", dRow, dCol);
#endif
}

void GEMM_BlockCyclicDecomposer::calculateTaskMap()
{
    /* Number of tiles needed from A and B to compute C tile. This should be equal */
    int numOfRows = A_Decomp.gridColumns;
    int numOfCols = B_Decomp.gridRows;

    /* Distribute tasks to devices */
    for (int i = 0; i < dRow; i++) {
        for (int j = 0; j < dCol; j++) {
            for (int k = 0; k < cTilesPerDevice; k++) {
                Task task;

                int taskNumber = i*dCol*cTilesPerDevice + j*cTilesPerDevice + k;

                int rowIndex = taskNumber/C_Decomp.gridColumns;
                int colIndex = taskNumber%C_Decomp.gridColumns;
                
                task.cTile = C_Decomp.tileMap[rowIndex][colIndex];
                task.aTiles = (Tile*) malloc(sizeof(Tile) * numOfRows);
                task.bTiles = (Tile*) malloc(sizeof(Tile) * numOfCols);

                int rowId = task.cTile.rowId;
                int colId = task.cTile.colId;

                for (int rows = 0; rows < numOfRows; rows++) {                            
                    task.aTiles[rows] = A_Decomp.tileMap[rowId][rows];
                }

                for (int cols = 0; cols < numOfCols; cols++) {
                    task.bTiles[cols] = B_Decomp.tileMap[cols][colId];
                }
                task.numOfTiles = numOfRows;
                taskMap[i*dCol + j][k] = task;
            }
        }
    }
}

void GEMM_BlockCyclicDecomposer::printTaskMap()
{
    for (int i = 0; i < numberOfDevices; i++) {
        std::cout << "Device: " << i << std::endl;
        for (int j = 0; j < cTilesPerDevice; j++) {
            printTask(&taskMap[i][j]);
        }
    }
    return; 
}

GEMM_BlockCyclicDecomposer::~GEMM_BlockCyclicDecomposer()
{
    for (int i = 0; i < numberOfDevices; i++) {
        free(taskMap[i]);
    }
    free(taskMap);
}

void printTile(Tile* tile) {
    printf("TILE ID: %d %d | ROWS: S-%d E-%d | COLUMNS: S-%d E-%d\n", tile->rowId, tile->colId, tile->sRow, tile->eRow, tile->sCol, tile->eCol);
}

void printTask(Task* task) {
    std::cout << "Tile C:" << std::endl;
    printTile(&task->cTile);
    std::cout << "Needed Tiles:" << std::endl;
    for (int i = 0; i < task->numOfTiles; i++) {
        printTile(&task->aTiles[i]);
        printTile(&task->bTiles[i]);
    }
    return;
}

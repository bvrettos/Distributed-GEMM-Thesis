#include "2DBlockCyclicDecomposition.hpp"

// #define DEBUG

BlockCyclicMatrixDecomposer::BlockCyclicMatrixDecomposer(int rows, int columns, int blockRows, int blockColumns) : matrixRows(rows), matrixColumns(columns), blockRows(blockRows), blockColumns(blockColumns)
{
        gridRows = (rows + blockRows - 1) / blockRows;
        gridColumns = (columns + blockColumns - 1) / blockColumns;

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

GEMM_BlockCyclicDecomposer::GEMM_BlockCyclicDecomposer(int M, int N, int K, int blockRows, int blockColumns, MPI_Comm problemCommunicator) : M(M), N(N), K(K), numberOfDevices(numberOfDevices), blockRows(blockRows), blockColumns(blockColumns), GEMM_Communicator(problemCommunicator), 
            A_Decomp(M, K, blockRows, blockColumns), B_Decomp(K, N, blockRows, blockColumns), C_Decomp(M, N, blockRows, blockColumns)
{
    MPI_Comm_size(GEMM_Communicator, &numberOfDevices);
    MPI_Comm_rank(GEMM_Communicator, &rank);
    communicatorSize = numberOfDevices;

    int cTiles = C_Decomp.size();
    cTilesPerDevice = cTiles / numberOfDevices;

    taskMap = (Task**) malloc(sizeof(Task*) * numberOfDevices);
    for (int i = 0; i < numberOfDevices; i++) {
        taskMap[i] = (Task*) malloc(sizeof(Task) * cTilesPerDevice);
    }

    /* Check if decomposition is completely square */
    if ((M == N) && (M == K) && (M % blockRows == 0) && (M % blockColumns) == 0) {
        squareDecomposition = true;
    }
    else
        squareDecomposition = false;

    calculateVirtualDeviceGrid();

    A_Decomp.decomposeMatrix();
    B_Decomp.decomposeMatrix();
    C_Decomp.decomposeMatrix();

    /* This should be the same as B_Decomp.gridColumns; */
    helperTilesPerTask = A_Decomp.gridColumns;
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

void GEMM_BlockCyclicDecomposer::scatterTasks()
{
    /* We have already created a task map in which we know the C tiles that each device calculates as well the dependencies for it */

    /* We need to translate the task into memory addresses for the node that sends the task to each process and then send them. */


}

void GEMM_BlockCyclicDecomposer::squareTaskScattering(double* A, double* B, double* C, double*** localA, double*** localB, double** localC)
{
    /* Start by distributing C. */
    MPI_Type_vector(blockRows, blockColumns, blockColumns, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &tile);
    MPI_Type_commit(&tile);

    MPI_Type_vector(blockRows, blockColumns, N, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &globalBlock);
    MPI_Type_commit(&globalBlock);

    scatterCountA = (int*) malloc(sizeof(int) * communicatorSize);
    scatterCountB = (int*) malloc(sizeof(int) * communicatorSize);
    
    /* Scatter Count is the same for everyone . */
    scatterCountC = (int*) malloc(sizeof(int) * communicatorSize);
    for (int i = 0; i < communicatorSize; i++) {
        scatterCountC[i] = 1;
    }

    /* Needs to be 2D since we need them for gathering as well... */
    scatterOffsetC = (int**) malloc(sizeof(int*) * cTilesPerDevice);
    for (int i = 0; i < cTilesPerDevice; i++) {
        scatterOffsetC[i] = (int *) malloc(sizeof(int) * communicatorSize);
    }

    /* Needs to be 2D because we want multiple arrays depending on tiles necessary for task calculation */
    scatterOffsetA = (int**) malloc(sizeof(int*) * helperTilesPerTask);
    scatterOffsetB = (int**) malloc(sizeof(int*) * helperTilesPerTask);

    for (int i = 0; i < helperTilesPerTask; i++) {
        scatterOffsetA[i] = (int*) malloc(sizeof(int*) * communicatorSize);
        scatterOffsetB[i] = (int*) malloc(sizeof(int*) * communicatorSize);
    }

    /* Create a set for A and B */
    std::map<int, int> aMap, bMap;

    double t1, t2, t3, scatterBandAtime = 0, scatterCtime=0;

    /* Completely square decomposition */
    for (int k = 0; k < cTilesPerDevice; k++) {
        for (int i = 0; i < dRow; i++) {
            for (int j = 0; j < dCol; j++) {
                    int skipIndexA = -1, skipIndexB = -1;
                    Tile cTile = taskMap[i*dCol + j][k].cTile;
                    Tile* aTiles = taskMap[i*dCol + j][k].aTiles;
                    Tile* bTiles = taskMap[i*dCol + j][k].bTiles;

                    int aRow = taskMap[i*dCol + j][k].aTiles[0].rowId;
                    int bCol = taskMap[i*dCol + j][k].bTiles[0].colId;

                    /* Scatter C */
                    scatterOffsetC[k][i*dCol + j] = cTile.colId*blockColumns + cTile.rowId*blockRows*N;

                    scatterCountA[i*dCol + j] = 0;
                    scatterCountB[i*dCol + j] = 0;

                    if (!findInMap(aMap, aRow, &skipIndexA))
                    {
                        aMap.insert({k, aRow});
                        scatterCountA[i*dCol + j] = 1;
                    }

                    if (!findInMap(bMap, bCol, &skipIndexB))
                    {
                        bMap.insert({k, bCol});
                        scatterCountB[i*dCol + j] = 1;
                    }
                    
                    for (int taskNum = 0; taskNum < helperTilesPerTask; taskNum++) {
                        scatterOffsetA[taskNum][i*dCol + j] = aTiles[taskNum].colId*blockColumns + aTiles[taskNum].rowId*blockRows*N;
                        scatterOffsetB[taskNum][i*dCol + j] = bTiles[taskNum].colId*blockColumns + bTiles[taskNum].rowId*blockRows*N;
                    }
            }
        }

        t1 = MPI_Wtime();
        /* Send to device */
        MPI_CHECK(MPI_Scatterv(C, scatterCountC, scatterOffsetC[k], globalBlock, localC[k], 1, tile, 0, GEMM_Communicator));
        t2 = MPI_Wtime();

        for (int taskNum = 0; taskNum < helperTilesPerTask; taskNum++) {
            MPI_CHECK(MPI_Scatterv(A, scatterCountA, scatterOffsetA[taskNum], globalBlock, localA[k][taskNum], 1, tile,  0, GEMM_Communicator));
            MPI_CHECK(MPI_Scatterv(B, scatterCountB, scatterOffsetB[taskNum], globalBlock, localB[k][taskNum], 1, tile,  0, GEMM_Communicator));
        }
        std::cout << "here" << std::endl;
        t3 = MPI_Wtime();
        scatterCtime += t2-t1;
        scatterBandAtime += t3-t2;
    }

    printf("Rank: %d finished scattering\n", rank);
    
    if (rank == 0)
        printf("Scatter B and A: %lf, Scatter C time: %lf\n", scatterBandAtime, scatterCtime);

    #ifdef DEBUG
        for (int i = 0; i < cTilesPerDevice; i++){
            writeMatrixToFile(localC[i], blockRows, blockColumns, "matrixC - rank" + std::to_string(rank));

            for (int taskNum = 0; taskNum < helperTilesPerTask; taskNum++) {
                writeMatrixToFile(localA[i][taskNum], blockRows, blockColumns, "matrixA - rank" + std::to_string(rank) + " Tile: " + std::to_string(i) +  " taskNum" + std::to_string(taskNum));
                writeMatrixToFile(localB[i][taskNum], blockRows, blockColumns, "matrixB - rank" + std::to_string(rank) + " Tile: " + std::to_string(i) + " taskNum" + std::to_string(taskNum));
            }
        }
    #endif

    return;
}

void GEMM_BlockCyclicDecomposer::squareTaskGathering(double* C, double** localC)
{
    for (int i = 0; i < cTilesPerDevice; i++) {
        MPI_CHECK(MPI_Gatherv(localC[i], 1, tile, C, scatterCountC, scatterOffsetC[i], globalBlock, 0, GEMM_Communicator));
    }
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
        std::cout << std::endl;
    }
    return;
}
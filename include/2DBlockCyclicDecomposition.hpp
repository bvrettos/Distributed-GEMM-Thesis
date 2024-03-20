#ifndef BLOCK_CYCLIC_DECOMPOSITION_HPP
#define BLOCK_CYCLIC_DECOMPOSITION_HPP

#include <iostream>
#include <cstdio>
#include <cmath>

typedef struct Tile {
    int sRow, eRow;
    int sCol, eCol;
    int rowId, colId;
} Tile; // Size: 6*4 = 24bytes;

typedef struct Task {
    Tile cTile;
    Tile *aTiles, *bTiles;
    int numOfTiles; // Number of tiles found in arrays;
} Task; // Size: 24 + 2*numOfTiles*24;

void printTile(Tile* tile);
void printTask(Task* task);

class BlockCyclicMatrixDecomposer
{
    public:
        int matrixRows, matrixColumns;  // Dimensions of Matrix
        int gridRows, gridColumns;      // Grid dimensions based on Tile size
        int blockRows, blockColumns;    // Tiling Dimensions
        char transpose;                 // Unused for now

        Tile** tileMap;                 // gridRows x gridColumns

        BlockCyclicMatrixDecomposer(int rows, int columns, int blockRows, int blockColumns);
        int size();
        void decomposeMatrix();
        void printTileMap();
        ~BlockCyclicMatrixDecomposer();
};

class GEMM_BlockCyclicDecomposer
{
    public:
        int M, N, K;                    // GEMM Problem Dimensions
        int numberOfDevices;            // ...
        int blockRows, blockColumns;    // Tiling Dimensions
        int cTilesPerDevice;            // How many tiles of C are sent to each device
        int dRow, dCol;                 // Virtual Device Grid dimensions (dRow x dCol)
    
        Task** taskMap;                 // dRow*dCol x cTilesPerDevice
        BlockCyclicMatrixDecomposer A_Decomp, B_Decomp, C_Decomp;
        
        GEMM_BlockCyclicDecomposer(int M, int N, int K, int numberOfDevices, int blockRows, int blockColumns);
        void calculateVirtualDeviceGrid();
        void calculateTaskMap();        // Currently, tasks are distributed kinda sequentially, but this can be changed by having more than one member functions
        void printTaskMap();

        void scatterTasks();
        ~GEMM_BlockCyclicDecomposer();
};

#endif
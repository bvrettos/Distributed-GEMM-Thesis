#ifndef BLOCK_CYCLIC_DECOMPOSITION_HPP
#define BLOCK_CYCLIC_DECOMPOSITION_HPP

#include <iostream>
#include <cstdio>
#include <cmath>
#include <mpi.h>
#include "errorHandling.hpp"
#include "cmatrix.h"


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
    
        Task** taskMap;                 // taskMap[dRow*dCol][cTilesPerDevice]
        BlockCyclicMatrixDecomposer A_Decomp, B_Decomp, C_Decomp;

        /* MPI Related Structures */
        MPI_Comm GEMM_Communicator;
        MPI_Datatype tileA, tileB, tileC; // If M = N = K, only one of them is needed since tilesize is the same
        MPI_Datatype globalBlockA, globalBlockB, globalBlockC; // Will see if the same is true for these ones
        MPI_Datatype dummy;             // Used for aligning MPI Datatypes

        int **scatterCountA, **scatterCountB, *scatterCountC;
        int **scatterOffsetC, ***scatterOffsetA, ***scatterOffsetB;
        int rank;
        int communicatorSize;
        int helperTilesPerTask;
        
        GEMM_BlockCyclicDecomposer(int M, int N, int K, int blockRows, int blockColumns, MPI_Comm problemCommunicator);
        void calculateVirtualDeviceGrid();
        void calculateTaskMap();        // Currently, tasks are distributed kinda sequentially, but this can be changed by having more than one member functions
        void printTaskMap();
        void scatterTasks();

        /* If Completely Square decomposition */
        bool squareDecomposition;
        MPI_Datatype tile, globalBlock;
        void squareTaskScattering(double* A, double* B, double* C, double*** localA, double*** localB, double** localC);
        void squareTaskGathering(double* C, double** localC);
        
        ~GEMM_BlockCyclicDecomposer();
};

#endif
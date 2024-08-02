#ifndef BLOCK_CYCLIC_DECOMPOSITION_HPP
#define BLOCK_CYCLIC_DECOMPOSITION_HPP

#include <iostream>
#include <cstdio>
#include <cmath>
#include <mpi.h>
#include "errorHandling.hpp"
#include "cmatrix.h"

class BlockCyclicMatrixDecomposer
{
    public:
        int M, N;                       // Dimensions of Matrix (rows x columns)
        int gridRows, gridColumns;      // Grid dimensions based on Tile size
        int blockRows, blockColumns;    // Tiling Dimensions
        char transpose;                 // Unused for now

        /* Process Grid (dRow x dCol) (Row Major)*/
        int dRow, dCol;
        int rank, size;
        int procRow, procCol;

        int rowDiv, rowMod;
        int colDiv, colMod;

        BlockCyclicMatrixDecomposer(int rows, int columns, int blockRows, int blockColumns, MPI_Comm communicator);
        BlockCyclicMatrixDecomposer(int rows, int columns, int blockRows, int blockColumns);

        void calculateVirtualDeviceGrid();
        ~BlockCyclicMatrixDecomposer();
};

#endif
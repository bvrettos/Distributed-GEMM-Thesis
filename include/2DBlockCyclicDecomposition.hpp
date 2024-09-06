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
    private:
        void calculateVirtualDeviceGrid();
        bool isTileMineScalapack(int tileRow, int tileColumn);

        /* Multiplication Algorithms */
        void multiplyCannon(); // Cannon Decomp
        void multiplySummaCyclic(); // Scalapack Decomp
        void mutliplySumma(); // Cannon Decomp

        /* Block Decompositions */
        void decomposeScalapack();
        void decomposeCannon();

    public:
        int M, N, K;                    // Dimensions of Matrix (rows x columns)
        int gridRows, gridColumns;      // Grid dimensions based on Tile size
        int blockRows, blockColumns;    // Tiling Dimensions
        char transpose;                 // Unused for now

        MPI_Comm rowCommunicator, columnCommunicator, worldCommunicator;

        /* Process Grid (dRow x dCol) (Row Major)*/
        int dRow, dCol;
        int rank, size;
        int procRow, procCol;

        int rowDiv, rowMod;
        int colDiv, colMod;
        DecompositionType decompositionType;

        BlockCyclicMatrixDecomposer(int rows, int columns, int blockRows, int blockColumns, MPI_Comm communicator);
        BlockCyclicMatrixDecomposer(int rows, int columns, int blockRows, int blockColumns);

        
        ~BlockCyclicMatrixDecomposer();
};

#endif
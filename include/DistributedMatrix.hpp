#ifndef DISTRIBUTED_MATRIX_HPP
#define DISTRIBUTED_MATRIX_HPP

#include <stdio.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <cassert>
#include <mpi.h>
#include <nccl.h>
#include <cmatrix.h>
#include <transfers.hpp>
#include <vector>
#include <generalUtilities.hpp>

#include <Tile.hpp>

/*
    Numroc returns the number of rows or columns
    of a distributed matrix assuming that this matrix was distributed
    over nprocs (used in order to calculate task size)
    
    n: number of rows/columns
    nb: blockRows/blockColumns
    iproc: process_row/process_col
    isrproc: rsrc/csrsc
    nprocs: dRow/dCol
*/

long long numroc(long long n, int nb, int iproc, int isproc, int nprocs);

/* In order to keep memory alive - we need to implement a function that transforms a global2local memory index (like ScaLAPACKS) */
int64_t local2global(int64_t local, int64_t nb, int iproc, int isrproc, int nprocs);
int64_t global2local(int64_t global, int64_t nb, int nprocs);

template <typename scalar_t>
class DistributedMatrix {
    private:
        long long rows, columns;            /* Global Matrix Dimensions */
        int blockRows, blockColumns;        /* Blocking Size (without Padding) */
        int dRow, dCol, dStack;             /* Right now dStack is unsued (used for 2.5D) */
        int processRow, processColumn, processStack;   // Communicator Grid 
        int rank, size;                      // Global Communicator info

        /* LAPACK Style Data */
        scalar_t* matrixData;                /* Actual data of matrix */
        long long ld, lld;                  /* Global - Local Leading Dimension */

        Tile<scalar_t>** tileMap;           /* Tile Map (will change it a bit) */
        int tileGridRows, tileGridColumns;          /* Dimensions of Tile Map*/
        bool rowModulo, columnModulo; // If dim%block != 0
        
        MPI_Comm mpiCommunicator;              /* Communicator used in distributing Matrix */
        ncclComm_t ncclCommunicator;
        MatrixLayout layout;                /* Row/Column Major */
        DistributionStrategy distribution;  /* Distribution Strategy - Defaults to 2D Block Cyclic */
        MemoryLocation memoryLocation;


        /* For Compatability Purposes - PBLAS info */
        // TODO

        void initializeTileMap();
        void calculate2DProcessGrid();
        void calculate3DProcessGrid();
    public:
        /* Empty Constructor */
        DistributedMatrix();   

        /* Device Information (Needs improvement) */
        int deviceId;
             

        /* Constructor with pointer passed to it - Used for ScaLAPACK and LAPACK wrappers */
        DistributedMatrix(long long rows, long long columns, scalar_t* A, int64_t ld, MemoryLocation location, MPI_Comm communicator, 
            MatrixLayout layout = MatrixLayout::ColumnMajor, DistributionStrategy distribution = DistributionStrategy::BlockCyclic);

        /* Constructor without blocking dimensions. Let system decide Distribution e.t.c. */
        DistributedMatrix(long long rows, long long columns, int64_t ld, MemoryLocation location, MPI_Comm communicator, 
            MatrixLayout layout = MatrixLayout::ColumnMajor, DistributionStrategy distribution = DistributionStrategy::BlockCyclic); 

        /* Constructor with blocking dimensions. Let user decide Distribution */
        DistributedMatrix(long long rows, long long columns, int64_t ld, int blockRows, int blockColumns,  MemoryLocation location, MPI_Comm communicator, 
            MatrixLayout layout = MatrixLayout::ColumnMajor); 
        
        /* Constructor with process grid dimensions as a parameter. */
        DistributedMatrix(long long rows, long long columns, int64_t ld, int blockRows, int blockColumns, int dRow, int dCol, MemoryLocation location, 
            MPI_Comm communicator, MatrixLayout layout = MatrixLayout::ColumnMajor); 

        /* Main Distribution Method (Los Pollos Hermanos) */
        void distribute(int rankRoot = 0);

        /* Gather Method - Gather matrix on LAPACK style pointer A */
        void gather(scalar_t* A, int64_t lda, int rankRoot = 0);

        /* 
            Wrappers for LAPACK & ScaLAPACK inputs. 
            - LAPACK Wrapper is used when a matrix exists only on one specific rank (root rank) and needs to be distributed to all other ranks.
            - ScaLAPACK Wrapper is used when a matrix is already 2D Block Cyclic distributed and need to be wrapped to be used by this Class.
        */

        /* Simple getters */
        int64_t getRows();
        int64_t getColumns();
        int gridRows();
        int gridColumns();
        int getDeviceId();
        void print();

        /* Tile Utilities */
        bool tileIsMine(int tileRow, int tileColumn);
        Tile<scalar_t>& getTile(int tileRow, int tileColumn);
};

template <typename scalar_t>
DistributedMatrix<scalar_t> fromLAPACK(int64_t m, int64_t n, scalar_t* A, int64_t lda, int64_t mb, int64_t nb, int rankRoot, MPI_Comm communicator, MemoryLocation location = MemoryLocation::Host);

template <typename scalar_t>
DistributedMatrix<scalar_t> fromScaLAPACK(int64_t m, int64_t n, scalar_t* A, int64_t lda, int64_t mb, int64_t nb, int p, int q, MPI_Comm communicator);

template <typename scalar_t>
DistributedMatrix<scalar_t> generateRandomMatrix(int64_t rows, int64_t columns, int blockRows, int blockColumns, MemoryLocation location, MPI_Comm communicator, MatrixLayout layout);

#endif
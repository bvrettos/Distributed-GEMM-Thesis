#ifndef TILE_HPP
#define TILE_HPP

#include <nccl.h>
#include <mpi.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <typeinfo>

/* Custom Includes */
#include <Enums.hpp> // Type Enumerators
#include <MatrixUtilities.hpp>

template <typename scalar_t>
class Tile {
    private:
        scalar_t* data;
        int64_t rows;
        int64_t columns;
        int64_t ld;
        MemoryLocation location;
        MatrixLayout layout;
        bool allocated;

        TileType tileType;      

    public:
        Tile(); // Empty Constructor
        Tile(int64_t rows, int64_t columns, MemoryLocation location, MatrixLayout layout = MatrixLayout::ColumnMajor); // No Input data constructor - but memory ready
        Tile(int64_t rows, int64_t columns, scalar_t* data, int64_t ld, MemoryLocation location, MatrixLayout layout = MatrixLayout::ColumnMajor); // Input Data Constructor
        Tile(Tile<scalar_t> sourceTile, scalar_t* data, int64_t ld); // Copy Data to existing Tile constructor

        /* Data Access Methods */
        int64_t getRows();
        int64_t getColumns();
        int64_t getStride();
        MatrixLayout getLayout();
        MemoryLocation getLocation();
        scalar_t* getDataPointer();

        /* Memory Management */
        void allocateMemory();
        bool isAllocated();

        /* MPI-Related Communication Functions */
        void send(int receiverRank, MPI_Comm communicator, int tag = 0);
        void isend(int receiverRank, MPI_Comm communicator, int tag, MPI_Request *request);
        void recv(int senderRank, MPI_Comm communicator, int tag = 0);
        void irecv(int senderRank, MPI_Comm communicator, int tag, MPI_Request *request);

        /* NCCL-Related Communication Functions */
        void send(int receiverRank, ncclComm_t communicator, cudaStream_t stream);
        void recv(int senderRank, ncclComm_t communicator, cudaStream_t stream);
        
        /* TODO: */
        // Transpose Function (execution either in CPU or GPU) - useful in gemm e.t.c.
        void transpose();
        void bcast(int broadcastRootRank, ncclComm_t communicator, cudaStream_t stream);
        // Random Tile Generation - useful for testing
        void generateRandomValues(int deviceLoc);

        /* Debug Stuff */
        void printTile(int rank);
        void writeTile(int tileRow, int tileColumn);

        // Add support for multiple devices per process - 
        // add a deviceId variable - Also add a rank variable for immediate access to owner (needed for SUMMA)
};



#endif
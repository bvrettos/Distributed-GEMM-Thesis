#ifndef DISTRIBUTED_MATRIX_HPP
#define DISTRIBUTED_MATRIX_HPP

#include <mpi.h>
#include "errorHandling.hpp"

typedef enum {
    2D_SEQUENTIAL,
    2D_CYCLIC,
    SCALAPACK
} distributionStrategy;

typedef enum {
    ROW_MAJOR,
    COL_MAJOR
} matrixOrder;


template <typename scalar_t>
class Tile {
    private:
        int rows, columns;
        int currentTileLocation; /* Check if tile is in device or host. Need to create a consistency protocol for updates. */
        int ld;

        int tileId, tileRow, tileColumn; // Identification of tile on distributedMatrix (not always needed)
        matrixOrder order;

        /* I can create a FSM for the runtime checks */
        bool allocated;
        bool toBeFreed;

        /* Actual data */
        scalar_t *data;
    public:
        /* Constructors */
        Tile();
        Tile(int rows, int columns);
        Tile(int rows, int columns, scalar_t* data)

        /* Copy Tile methods */


        /* Convert arrays/pointers to Tile methods */

        /* Send tile to host or device */

        /* Access pointers for Tiles */
        scalar_t* dataPointer();
};

class distributedMatrix {
    private:
        MPI_Comm matrixCommunicator;
        int dRow, dCol, numDevices;
        int M, N;
        int numTiles;
        std::string name; // For debugging purposes 
        distributionStrategy strategy;

        Tile* tileArray;

        Tile* temporaryTiles; /* Whatever tiles may be needed for calculation purposes */

        /* Data will be accessed through Tiles. Only Tiles local to each process can be accessed (without sending). */
        void createProcessGrid();
        void chooseDistributionStrategy();
        void scalapackDistribution();
        void distributeTiles(distributionStrategy strategy);

    public:
        distributedMatrix(int M, int N, MPI_Comm communicator); //Should determine tiling size by itself (need to model)
};


#endif
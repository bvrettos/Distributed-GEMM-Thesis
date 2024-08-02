#ifndef GRID_LAYOUT_HPP
#define GRID_LAYOUT_HPP

#include <mpi.h>

class Block {
    private:
        void *data;
        int ld;
        int rows;
        int columns;
        MPI_Datatype mpiDatatype;
    public:
        void* getDataPointer();
}

class GridLayout {
    private:
        int rowBlocks, columnBlocks;
        int* rowSplits, columnSplits;
        int** owners;

        int localBlocks;
        Block* localBlocks;
    public:
        void scatterMatrix(void* matrix);
};

#endif
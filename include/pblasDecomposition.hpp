#ifndef PBLAS_DECOMPOSITION_HPP
#define PBLAS_DECOMPOSITION_HPP

#include <mpi.h>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <cmatrix.h>
#include <unistd.h>
#include <mpi-ext.h>
#include <cudaCommunicator.hpp>
#include <transfers.hpp>

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
int numroc(int n, int nb, int iproc, int isrproc, int nprocs);

class pblasDecomposer
{
    public:
        int M, N; // Matrix Dimensions
        int Mb, Nb; // Tiling Factors 
        int dRow, dCol; // Process-Grid dimensions
        int rank, size; // MPI-Information
        int procRow, procColumn; // Process-Grid IDs 
        int myBlocks;

        MPI_Comm communicator;
        /* Results of Numroc | Dimensions of local matrix */
        int localRows, localColumns;
        bool cudaAwareMPI;

        int gridRows, gridColumns;

    pblasDecomposer(int M, int N, int Mb, int Nb, MPI_Comm communicator);
    ~pblasDecomposer();
    void scatterMatrix(int senderRank, double* globalMatrix, double* localMatrix);
    void gatherMatrix(int receiverRank, double* globalMatrix, double* localMatrix);
};

#endif
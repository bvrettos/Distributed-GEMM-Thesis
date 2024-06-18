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

extern "C" {
    /* Cblacs declarations */
    void Cblacs_pinfo(int*, int*);
    void Cblacs_get(int, int, int*);
    void Cblacs_gridinit(int*, const char*, int, int);
    void Cblacs_gridinfo(int, int*, int*, int*, int*);
    void Cblacs_pcoord(int, int, int*, int*);
    void Cblacs_gridexit(int);
    void Cblacs_barrier(int, const char*);
    void Cdgerv2d(int, int, int, double*, int, int, int);
    void Cdgesd2d(int, int, int, double*, int, int, int);

    void pdgemr2d_(const int *m, const int *n, const double *a, const int *ia, const int *ja, const int *desca,
               double *b, const int *ib, const int *jb, const int *descb, const int *ictxt);

    void descinit_(int *desc, const int *m,  const int *n, const int *mb, 
    const int *nb, const int *irsrc, const int *icsrc, const int *ictxt, 
    const int *lld, int *info);

}
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
        int M, N;
        int Mb, Nb;
        int dRow, dCol;

        /* CBLACS Related Stuf*/
        int rank, numberOfProcesses;
        int procRow, procCol;
        int cblacsContext; 

        int allCblacsContext, rootCblacsContext;
        int rootRow, rootCol;
        int localMatrixDescriptor[9], globalMatrixDescriptor[9];
        MPI_Comm communicator;

        /* Results of Numroc | Dimensions of local matrix */
        int localRows, localColumns;

        double *localMatrix;    // Allocated inside
        double *globalMatrix;   // Passed in constructor

        bool cudaAwareMPI;

    pblasDecomposer(int M, int N, int Mb, int Nb, int dRow, int dCol, double* globalMatrix, MPI_Comm communicator);
    ~pblasDecomposer();

    void allocateLocalMatrix();
    void createCblacsContext();
    void scalapackScatter();
    void scalapackGather();

    /* Maybe deprecated */
    void scatterMatrix();
    void gatherMatrix();
};

class scalapackDecomposer
{
    public:
        int M, N, K;
        int Mb, Nb;
        int dRow, dCol;

        int rank, size;
        int processRow, processColumn;

        int cblacsContext; 
        int allCblacsContext, rootCblacsContext;

        pblasDecomposer A, B, C;
};

#endif
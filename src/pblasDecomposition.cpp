#include "pblasDecomposition.hpp"
#define DEBUG

int numroc(int n, int nb, int iproc, int isrproc, int nprocs)
{
    int extraBlocks, myDistance, numBlocks;
    int numroc = 0;

    myDistance = (nprocs + iproc - isrproc) % nprocs;

    numBlocks = n / nb;
    numroc = (numBlocks/nprocs) * nb;

    extraBlocks = numBlocks % nprocs;

    if (myDistance < extraBlocks) {
        numroc += nb;
    }
    else if (myDistance == extraBlocks) {
        numroc += n % nb;
    }

    return numroc;
}

pblasDecomposer::pblasDecomposer(int M, int N, int Mb, int Nb, int dRow, int dCol, double* globalMatrix, MPI_Comm communicator) : M(M), N(N), Mb(Mb), Nb(Nb)
{
    this->communicator = communicator;
    this->globalMatrix = globalMatrix;

    /* Calculate grid dimensions */
    this->dRow = dRow;
    this->dCol = dCol;

    rootRow = 0;
    rootCol = 0;

    scatterMatrix();
}

void pblasDecomposer::createCblacsContext()
{   
    Cblacs_pinfo(&rank, &numberOfProcesses);  
    Cblacs_get(0, 0, &cblacsContext);
    Cblacs_gridinit(&cblacsContext, "Row-Major", dRow, dCol);
    Cblacs_pcoord(cblacsContext, rank, &procRow, &procCol);

    #ifdef DEBUG
        for (int row = 0; row < dRow; row++) {
            for (int col = 0; col < dCol; col++) {
                if (procRow == row && procCol == col)
                    printf("Rank: %d, procRow: %d, procCol: %d\n", rank, procRow, procCol);
            }
            if (rank == 0)
                std::cout << std::endl;
        }
    #endif
}

void pblasDecomposer::allocateLocalMatrix()
{
    localRows = numroc(M, Mb, procRow, rootCol, dRow);
    localColumns = numroc(N, Nb, procCol, rootRow, dCol);

    localMatrix = (double* ) malloc(sizeof(double) * localRows * localColumns);

    #ifdef DEBUG
        printf("Rank: %d, Local Rows: %d, Local Columns: %d\n", rank, localRows, localColumns);
    #endif
}

void pblasDecomposer::scatterMatrix()
{
    createCblacsContext();
    allocateLocalMatrix();

    int sendRow = 0, sendCol = 0;
    int receiveRow=0, receiveColumn=0;
    MPI_Datatype datatype, globalDatatype;

    bool vertical, horizontal;

    for (int row = 0; row < M; row += Mb, sendRow = (sendRow+1) % dRow) {
        sendCol = 0;
        horizontal = false;

        int numRows = Mb;
        /* Is this last row block?*/
        if (M - row < Mb)
            numRows = M - row;
            
        for (int col = 0; col < N; col += Nb, sendCol = (sendCol+1) % dCol) {
            int numCols = Nb;
            /* Is this last column block? */
            if (N - col < Nb)
                numCols = N - col;

            /* Send block*/
            if (rank == 0) {            
                Cdgesd2d(cblacsContext, numRows, numCols, &globalMatrix[col*M + row], M, sendRow, sendCol);
            }

            /* Receive block*/
            if (procRow == sendRow && procCol == sendCol) {
                Cdgerv2d(cblacsContext, numRows, numCols, &localMatrix[receiveColumn*localRows + receiveRow], localRows, 0, 0);
                receiveColumn = (receiveColumn + numCols) % localColumns;
            }
        }

        if (procRow == sendRow) 
            receiveRow = (receiveRow + numRows) % localRows;
        Cblacs_barrier(cblacsContext, "All");
    }

    #ifdef DEBUG
        printf("Rank: %d finished scattering data\n", rank);
    #endif
}

void pblasDecomposer::gatherMatrix()
{
    /* Gather matrix */
    int sendRow = 0, sendCol = 0;
    int receiveRow=0, receiveColumn=0;

    for (int row = 0; row < M; row += Mb, sendRow=(sendRow+1) % dRow) {
        sendCol = 0;
        // Number of rows to be sent
        // Is this the last row block?
        int numRows = Mb;
        if (M - row < Mb)
            numRows = M - row;
 
        for (int col = 0; col < N; col += Nb, sendCol = (sendCol+1) % dCol) {
            // Number of cols to be sent
            // Is this the last col block?
            int numCols = Nb;
            if (N - col < Nb)
                numCols = N - col;
 
            if (procRow == sendRow && procCol == sendCol) {
                // Send a nr-by-nc submatrix to process (sendr, sendc)
                Cdgesd2d(cblacsContext, numRows, numCols, &localMatrix[receiveColumn*localRows + receiveRow], localRows, 0, 0);
                receiveColumn = (receiveColumn + numCols) % localColumns;
            }
 
            if (rank == 0) {
                // Receive the same data
                // The leading dimension of the local matrix is nrows!
                Cdgerv2d(cblacsContext, numRows, numCols, &globalMatrix[col*M + row], M, sendRow, sendCol);
            }
        }
 
        if (procRow == sendRow) 
            receiveRow = (receiveRow + numRows) % localRows;

        Cblacs_barrier(cblacsContext, "All");
    }
}

pblasDecomposer::~pblasDecomposer()
{
    free(localMatrix);
    Cblacs_gridexit(cblacsContext);
}
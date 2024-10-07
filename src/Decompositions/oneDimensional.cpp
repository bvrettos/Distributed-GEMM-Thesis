#include <decompositions.hpp>

/* 
    Includes code for One Dimensional matrix decompositions.
*/


/* 1D Sequential Row Decomposition */
void sequentialRowDecomposition(long long M, long long N, int dRow, int dCol, bool columnMajor)
{
    /* If M%dRow == 0 -> You can scatter. If not, then send block iteratively */
    long long ld = N; // Row-Major
    if (columnMajor) // Column-Major
        ld = M;
    
    int processSize = dRow*dCol;
    /* Get rank */

    /* */
    int* scatterValues = new int[processSize];
    int* scatterOffsets = new int[processSize];

    for (int i = 0; i < dRow; i++) {
        
    }


}

/* 1D Block Column Decomposition */
void sequentialColumnDecomposition()
{

}


/* 1D Row Cyclic Decomposition */

/* 1D Column Cyclic Decomposition*/
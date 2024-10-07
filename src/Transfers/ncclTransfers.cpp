#include <transfers.hpp>

/* NCCL does not support strided - 2D Transfers, we need to first cut the arrays into 1D segments and then transfer */

void transferBlockNCCL(long long rows, long long columns, void* sourcePointer, const long long stride, const int destinationRank, int tag, cudaStream_t cudaStream, bool colMajor)
{
    if (colMajor) {
        std::swap(rows, columns);
    }

    if (sourcePointer == NULL) {
        /* Exit */
    }

    

}
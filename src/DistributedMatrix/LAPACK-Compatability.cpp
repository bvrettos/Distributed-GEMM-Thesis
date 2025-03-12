#include <DistributedMatrix.hpp>

// template <typename scalar_t>
// DistributedMatrix<scalar_t> DistributedMatrix<scalar_t>::fromLAPACK(int64_t m, int64_t n, scalar_t* A, int64_t lda, int64_t mb, int64_t nb, int rankRoot, MPI_Comm communicator, MemoryLocation location)
// {
//     // DistributedMatrix<scalar_t> matrix(m, n, lda, mb, nb, location, communicator, MatrixLayout::ColumnMajor, DistributionStrategy::BlockCyclic);
    
//     // /* Scatter Matrix */
//     // matrix.distribute(rankRoot);

//     // return matrix;
// }

// template <typename scalar_t>
// DistributedMatrix<scalar_t> DistributedMatrix<scalar_t>::fromScaLAPACK(int64_t m, int64_t n, scalar_t* A, int64_t lda, int64_t mb, int64_t nb, int p, int q, MPI_Comm communicator)
// {
//     // DistributedMatrix<scalar_t> matrix(m, n, lda, mb, nb,  )

//     // return matrix;
// }

long long numroc(int n, int nb, int iproc, int isrproc, int nprocs)
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

int64_t global2local(int64_t global, int64_t nb, int nprocs)
{
    return (global/(nb*nprocs))*nb + (global % nb);
}

int64_t local2global(int64_t local, int64_t nb, int iproc, int isrcproc, int nprocs)
{
    return (nprocs*(local/nb) + (nprocs + iproc - isrcproc) % nprocs)*nb + (local % nb);
}


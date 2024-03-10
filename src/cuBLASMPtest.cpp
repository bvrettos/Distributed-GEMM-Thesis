#include "cuBLASMP_wrappers.hpp"

int main(int argc, char* argv[])
{
    // Initialize MPI, create some distribute data
    MPI_Init(NULL, NULL);
    int rank,size;

    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &size);

    MPI_Finalize();

    return 0;
}
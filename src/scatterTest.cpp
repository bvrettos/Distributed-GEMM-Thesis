#include <mpi.h>
#include "cmatrix.h"

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int M, N;
    M = 16;
    N = 16;

    double *a;
    if (rank == 0) {
        a = malloc(sizeof(double)* M * N);
        generateMatrix(a, M, N);
    }

    MPI_Datatype dummy, localA, globalA;
    MPI_Type_vector(M, N, N, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &globalA);
    MPI_Type_commit(&globalA);

    MPI_Type_vector(M/4, N/4, N, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &localA);
    MPI_Type_commit(&localA);
}
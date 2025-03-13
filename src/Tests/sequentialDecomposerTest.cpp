#include "2DBlockSequentialDecomposition.hpp"
 

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long M = atoll(argv[1]);
    long long N = atoll(argv[2]);
    long long K = atoll(argv[3]);

    BlockSequentialDecomposer decomposer(M, N, K, MPI_COMM_WORLD);

    /* Test copying of matrices */
    double *A, *B, *C;
        if (rank == 0) {
        A = (double*) malloc(sizeof(double) * M * K);
        B = (double*) malloc(sizeof(double) * N * K);
        C = (double*) malloc(sizeof(double) * M * N);
        MatrixInit(A, M, K, 0);
        MatrixInit(B, K, N, 0);
        MatrixInit(C, M, N, 0);    
        printMatrixColumnMajor(C, M, K, 0);
    }
    /* Allocate local memory */
    long long localM, localK, localN;
    double *localA, *localB, *localC;
    localM = decomposer.localM;
    localN = decomposer.localN;
    localK = decomposer.localK;

    /* Debug decomposer */
    decomposer.deliverMatrix(A, B, C, &localA, &localB, &localC);

    decomposer.gatherResult(0, C, localC);

    MPI_Finalize();
    
    return 0;
}
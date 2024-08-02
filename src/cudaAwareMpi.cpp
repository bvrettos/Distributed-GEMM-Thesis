#include <cuda.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
// #include <mpi-ext.h>
#include <errorHandling.hpp>

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int size, rank;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int devId = 0;
    CUDA_CHECK(cudaSetDevice(devId));

    double* buffer;
    int length = atoi(argv[1]);
    CUDA_CHECK(cudaMallocManaged((void**)&buffer, sizeof(double) * length));
    double* hostBuffer;
    hostBuffer = (double*) malloc(sizeof(double) * length);
    for (int i = 0; i < length; i++)
        hostBuffer[i] = (double) i;


    CUDA_CHECK(cudaMemcpy(buffer, hostBuffer, length, cudaMemcpyHostToDevice));
    /* Copy deviceBuffer directly to other rank */
    double* newBuffer;
    if (rank == 0) {
        MPI_Send(&buffer[0], length, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
    }
    else if (rank == 1) {
        CUDA_CHECK(cudaMallocManaged((void**)&newBuffer, sizeof(double) * length));
        MPI_Recv(&newBuffer[0], length, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, NULL);
    }

    if (rank == 1) {
        printf("New Buffer\n");
        CUDA_CHECK(cudaMemcpy(newBuffer, buffer, length, cudaMemcpyDeviceToHost));
        for (int i = 0; i < length; i++) {
            printf("%lf ", buffer[i]);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
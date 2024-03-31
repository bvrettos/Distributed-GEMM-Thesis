#include <cmatrix.h>

__device__ double device_dabs(double value)
{
    if (value >= 0) return value;
    else return -value;
}

__device__ long int get_tid()
{
    return blockDim.x * blockIdx.x + threadIdx.x;
}

__global__ void initializeRandom(curandState *state, unsigned long seed)
{
    long int tid = get_tid();
    curand_init(seed, tid, 0, &state[tid]);
    return;
}

__global__ void generateMatrix(double* array, int rows, int columns, curandState *state)
{
    /* Max array value is (rows-1)*columns + columns-1 */
    long int tid = get_tid();

    /* Bounds check */
    if (tid > (rows-1)*columns + columns-1) return;

    array[tid] = device_dabs(curand_normal_double(&state[tid]));
    return;
}

void generateMatrixGPU(double* array, const int rows, const int columns)
{
    cudaSetDevice(0);

    curandState *devStates;
    long long int size = rows*columns;

    CUDA_CHECK(cudaMalloc((void **)&devStates, rows * columns * sizeof(curandState)));
    initializeRandom<<<1, size>>>(devStates, time(NULL));

    double* matrix;
    CUDA_CHECK(cudaMalloc((void **)&matrix, rows * columns * sizeof(double)));

    generateMatrix<<<1, size>>>(matrix, rows, columns, devStates);

    CUDA_CHECK(cudaMemcpy(array, matrix, rows*columns*sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(devStates));
    CUDA_CHECK(cudaFree(matrix));
}
#include <transfers.hpp>

cudaMemcpyKind findMemcpyKind(const void* sourcePointer, const void* destinationPointer)
{
    /* By default, make this a host-to-host copy */
    cudaMemcpyKind copyDirection = cudaMemcpyHostToHost;

    cudaPointerAttributes sourceAttributes, destinationAttributes;
    CUDA_CHECK(cudaPointerGetAttributes(&sourceAttributes, sourcePointer));
    CUDA_CHECK(cudaPointerGetAttributes(&destinationAttributes, destinationPointer));

    if (sourceAttributes.type == cudaMemoryTypeDevice && destinationAttributes.type == cudaMemoryTypeDevice)
        copyDirection = cudaMemcpyDeviceToDevice;
    else if (sourceAttributes.type == cudaMemoryTypeHost && destinationAttributes.type == cudaMemoryTypeDevice)
        copyDirection = cudaMemcpyHostToDevice;
    else if (sourceAttributes.type == cudaMemoryTypeDevice && destinationAttributes.type == cudaMemoryTypeHost)
        copyDirection = cudaMemcpyDeviceToHost;

    return copyDirection;   
}

template <typename T>
void copy(size_t size, const T* sourcePointer, T* destinationPointer, cudaMemcpyKind cudaCopyDirection)
{
    static_assert(std::is_trivially_copyable<T>(), "Elements must be copyable by C standards\n");
    cudaMemcpy(destinationPointer, sourcePointer, sizeof(T) * size, cudaCopyDirection);
}

template <typename T>
void copyBlock(long long rows, long long columns, T* sourcePointer, T* destinationPointer, 
    const long long ldSource, const long long ldDestination, bool columnMajor)
{
    if (sourcePointer == NULL || destinationPointer == NULL) {
        printf("On copyBlock: Pointers not initialized correctly\n");
        return;
    }

    int blockSize = rows * columns;
    if (blockSize <= 0) return;

    cudaMemcpyKind copyDirect = findMemcpyKind(sourcePointer, destinationPointer);
    
    if (!columnMajor)
        std::swap(rows, columns);

    #ifdef DEBUG
        printf("Memcpy Kind: %d\n", copyDirection);
    #endif

    /* If not strided, you can copy whole 2D block in a single go */
    if (rows == (size_t) ldSource && columns == (size_t) ldDestination) {
        copy(blockSize, sourcePointer, destinationPointer, copyDirection);
    }
    /* Else, copy column by column */
    else {
        #pragma GCC ivdep
        #pragma GCC unroll 32
        for (size_t cols = 0; cols < columns; cols++) {
            copy(rows, sourcePointer + (ldSource*cols), destinationPointer + (ldDestination*cols), copyDirection);
        }
    }
    return;
}

void memcpy2D(void* sourcePointer, long long ldSource, void* destinationPointer, long long ldDestination, long long rows, long long columns, size_t elementSize, bool colMajor)
{

    /* Check if pointers are valid */
    if (sourcePointer == NULL || destinationPointer == NULL) {
        printf("On memcpy2D: Pointers not initialized correctly\n");
        return;
    }

    if (!columnMajor)
        std::swap(rows, columns);

    cudaMemcpyKind = findMemcpyKind(sourcePointer, destinationPointer);
    CUDA_CHECK(cudaMemcpy2D(destinationPointer, ldDestination*elementSize, sourcePointer, ldSource*elementSize, rows*elementSize, columns, cudaMemcpyKind));

    return;
}


template void copy<double>(size_t size, const double* sourcePointer, double* destinationPointer, cudaMemcpyKind cudaCopyDirection);
template void copy<float>(size_t size, const float* sourcePointer, float* destinationPointer, cudaMemcpyKind cudaCopyDirection);

template void copyBlock<double>(long long rows, long long columns, double* sourcePointer, double* destinationPointer, const long long ldSource, const long long ldDestination, bool columnMajor);
template void copyBlock<float>(long long rows, long long columns, float* sourcePointer, float* destinationPointer, const long long ldSource, const long long ldDestination, bool columnMajor);

template void transferBlock<float>(long long rows, long long columns, float* sourcePointer, const long long stride, const int destinationRank, int tag, MPI_Request* requestHandle, bool colMajor);
template void transferBlock<double>(long long rows, long long columns, double* sourcePointer, const long long stride, const int destinationRank, int tag, MPI_Request* requestHandle, bool colMajor);

template void receiveBlock<float>(long long rows, long long columns, float* destinationPointer, const long long stride, const int sourceRank, int tag, MPI_Request* requestHandle, bool colMajor);
template void receiveBlock<double>(long long rows, long long columns, double* destinationPointer, const long long stride, const int sourceRank, int tag, MPI_Request* requestHandle, bool colMajor);
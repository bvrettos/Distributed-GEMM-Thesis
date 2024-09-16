#include <transfers.hpp>

// #define DEBUG

std::map<std::tuple<long, long, long>, MPI_Datatype*> datatypeCache;

/* 
    A process could find a datatype in the cache while another process can not find it. Datatypes are not tied to processes, this could be
    a huge mistake, I need to study it. However, I will still work with it for now.
*/
bool findDatatypeInCache(long long rows, long long columns, long long stride)
{
    std::map<std::tuple<long, long, long>, MPI_Datatype*>::iterator it = datatypeCache.find({rows, columns, stride});
    return (it != datatypeCache.end());
}

/* Point-to-Point Communication */
template <typename scalar_t>
void transferBlock(long long rows, long long columns, scalar_t* sourcePointer, const long long stride, const int destinationRank, int tag, MPI_Request* requestHandle, bool colMajor)
{
    MPI_Datatype scalarType;
    if (typeid(scalar_t) == typeid(float)) scalarType = MPI_FLOAT;
    else if (typeid(scalar_t) == typeid(double)) scalarType = MPI_DOUBLE;

    if (colMajor) {
        std::swap(rows, columns);
    }

    // MPI_Datatype* block;
    MPI_Datatype block;
    // if (!findDatatypeInCache(rows, columns, stride)) {
    MPI_Type_vector(rows, columns, stride, scalarType, &block);
    MPI_Type_commit(&block);
    // datatypeCache.at({rows, columns,stride}) = block;
    // }
    // else
    //     block = datatypeCache.at({rows, columns, stride});

    MPI_CHECK(MPI_Isend(sourcePointer, 1, block, destinationRank, tag, MPI_COMM_WORLD, requestHandle));
    return;
}

template <typename scalar_t>
void receiveBlock(long long rows, long long columns, scalar_t* destinationPointer, const long long stride, const int sourceRank, int tag, MPI_Request* requestHandle, bool colMajor)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Datatype scalarType;
    if (typeid(scalar_t) == typeid(float)) scalarType = MPI_FLOAT;
    else if (typeid(scalar_t) == typeid(double)) scalarType = MPI_DOUBLE;

    if (colMajor) {
        std::swap(rows, columns);
    }

    // MPI_Datatype* block;
    MPI_Datatype block;
    // if (!findDatatypeInCache(rows, columns, stride)) {
    MPI_Type_vector(rows, columns, stride, scalarType, &block);
    MPI_Type_commit(&block);
        // datatypeCache.at({rows, columns,stride}) = block;
    // }
    // else
    //     block = datatypeCache.at({rows, columns, stride});
    
    /* TODO: Check if destination is managed memory, or else CUDA Aware MPI will not work and you have to manually copy from device to host and then device */
    MPI_CHECK(MPI_Irecv(destinationPointer, 1, block, sourceRank, tag, MPI_COMM_WORLD, requestHandle));
    return;
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


template void copy<double>(size_t size, const double* sourcePointer, double* destinationPointer, cudaMemcpyKind cudaCopyDirection);
template void copy<float>(size_t size, const float* sourcePointer, float* destinationPointer, cudaMemcpyKind cudaCopyDirection);

template void copyBlock<double>(long long rows, long long columns, double* sourcePointer, double* destinationPointer, const long long ldSource, const long long ldDestination, bool columnMajor);
template void copyBlock<float>(long long rows, long long columns, float* sourcePointer, float* destinationPointer, const long long ldSource, const long long ldDestination, bool columnMajor);

template void transferBlock<float>(long long rows, long long columns, float* sourcePointer, const long long stride, const int destinationRank, int tag, MPI_Request* requestHandle, bool colMajor);
template void transferBlock<double>(long long rows, long long columns, double* sourcePointer, const long long stride, const int destinationRank, int tag, MPI_Request* requestHandle, bool colMajor);

template void receiveBlock<float>(long long rows, long long columns, float* destinationPointer, const long long stride, const int sourceRank, int tag, MPI_Request* requestHandle, bool colMajor);
template void receiveBlock<double>(long long rows, long long columns, double* destinationPointer, const long long stride, const int sourceRank, int tag, MPI_Request* requestHandle, bool colMajor);
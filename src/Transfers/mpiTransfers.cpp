#include <transfers.hpp>

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

    MPI_Datatype block;
    
    MPI_Type_vector(rows, columns, stride, scalarType, &block);
    MPI_Type_commit(&block);


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

    MPI_Datatype block;
    MPI_Type_vector(rows, columns, stride, scalarType, &block);
    MPI_Type_commit(&block);
    
    /* TODO: Check if destination is managed memory, or else CUDA Aware MPI will not work and you have to manually copy from device to host and then device */
    MPI_CHECK(MPI_Irecv(destinationPointer, 1, block, sourceRank, tag, MPI_COMM_WORLD, requestHandle));
    return;
}


template void transferBlock<float>(long long rows, long long columns, float* sourcePointer, const long long stride, const int destinationRank, int tag, MPI_Request* requestHandle, bool colMajor);
template void transferBlock<double>(long long rows, long long columns, double* sourcePointer, const long long stride, const int destinationRank, int tag, MPI_Request* requestHandle, bool colMajor);

template void receiveBlock<float>(long long rows, long long columns, float* destinationPointer, const long long stride, const int sourceRank, int tag, MPI_Request* requestHandle, bool colMajor);
template void receiveBlock<double>(long long rows, long long columns, double* destinationPointer, const long long stride, const int sourceRank, int tag, MPI_Request* requestHandle, bool colMajor);
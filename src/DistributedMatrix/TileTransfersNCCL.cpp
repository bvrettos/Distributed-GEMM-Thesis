#include <Tile.hpp>

template <typename scalar_t>
void Tile<scalar_t>::send(int receiverRank, ncclComm_t communicator, cudaStream_t stream)
{
    /* Check that memory is actually in device, else abort */
    if (location != MemoryLocation::Device) {
        /* abort */
    }
        
    ncclDataType_t scalarType;
    
    if (typeid(scalar_t) == typeid(float)) scalarType = ncclFloat;
    else if (typeid(scalar_t) == typeid(double)) scalarType = ncclDouble;

    ncclSend(this->data, this->rows*this->columns, scalarType, receiverRank, communicator, stream);
    return;
}

template <typename scalar_t>
void Tile<scalar_t>::recv(int senderRank, ncclComm_t communicator, cudaStream_t stream)
{
    /* Check that memory is actually in device, else abort */
    if (location != MemoryLocation::Device) {
        
    }

    ncclDataType_t scalarType;
    if (typeid(scalar_t) == typeid(float)) scalarType = ncclFloat;
    else if (typeid(scalar_t) == typeid(double)) scalarType = ncclDouble;

    ncclRecv(this->data, this->rows*this->columns, scalarType, senderRank, communicator, stream);

    return;
}

template <typename scalar_t>
void Tile<scalar_t>::bcast(int broadcastRootRank, ncclComm_t communicator, cudaStream_t stream)
{
    /* Check that memory is actually in device, else abort */
    if (location != MemoryLocation::Device) {
        
    }

    ncclDataType_t scalarType;
    if (typeid(scalar_t) == typeid(float)) scalarType = ncclFloat;
    else if (typeid(scalar_t) == typeid(double)) scalarType = ncclDouble;

    /* First find out the size. This depends on RootRanks dimensions */
    // ncclBroadcast(this->data, this->data, this->rows*this->columns)

    return;
}
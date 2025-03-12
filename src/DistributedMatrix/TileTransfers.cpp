#include <Tile.hpp>

template <typename scalar_t>
void Tile<scalar_t>::send(int receiverRank, MPI_Comm communicator, int tag)
{
    MPI_Datatype scalarType;
    if (typeid(scalar_t) == typeid(float)) scalarType = MPI_FLOAT;
    else if (typeid(scalar_t) == typeid(double)) scalarType = MPI_DOUBLE;

    MPI_Send(this->data, this->rows*this->columns, scalarType, receiverRank, tag, communicator);

    return;
}

template <typename scalar_t>
void Tile<scalar_t>::isend(int receiverRank, MPI_Comm communicator, int tag, MPI_Request *request)
{
    MPI_Datatype scalarType;
    if (typeid(scalar_t) == typeid(float)) scalarType = MPI_FLOAT;
    else if (typeid(scalar_t) == typeid(double)) scalarType = MPI_DOUBLE;

    MPI_Send(this->data, this->rows*this->columns, scalarType, receiverRank, tag, communicator);

    return;
}

template <typename scalar_t>
void Tile<scalar_t>::recv(int senderRank, MPI_Comm communicator, int tag)
{
    MPI_Datatype scalarType;
    if (typeid(scalar_t) == typeid(float)) scalarType = MPI_FLOAT;
    else if (typeid(scalar_t) == typeid(double)) scalarType = MPI_DOUBLE;

    MPI_Recv(this->data, this->rows*this->columns, scalarType, senderRank, tag, communicator, MPI_STATUS_IGNORE);

    return;
}

template <typename scalar_t>
void Tile<scalar_t>::irecv(int senderRank, MPI_Comm communicator, int tag, MPI_Request* request)
{
    MPI_Datatype scalarType;
    if (typeid(scalar_t) == typeid(float)) scalarType = MPI_FLOAT;
    else if (typeid(scalar_t) == typeid(double)) scalarType = MPI_DOUBLE;

    MPI_Recv(this->data, this->rows*this->columns, scalarType, senderRank, tag, communicator, request);

    return;
}
#include <calSetup.hpp>

static calError_t allgather(void* src_buf, void* recv_buf, size_t size, void* data, void** request)
{
    MPI_Request req;
    int err = MPI_Iallgather(src_buf, size, MPI_BYTE, recv_buf, size, MPI_BYTE, (MPI_Comm)(data), &req);
    if (err != MPI_SUCCESS)
    {
        return CAL_ERROR;
    }
    *request = (void*)(req);
    return CAL_OK;
}

static calError_t request_test(void* request)
{
    MPI_Request req = (MPI_Request)(request);
    int completed;
    int err = MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
    if (err != MPI_SUCCESS)
    {
        return CAL_ERROR;
    }
    return completed ? CAL_OK : CAL_ERROR_INPROGRESS;
}

static calError_t request_free(void* request)
{
    return CAL_OK;
}

cal_comm_t createCalCommunicator(int rank, int size, int localDeviceID)
{   
    cal_comm_t calCommunicator = NULL;

    cal_comm_create_params_t parameters;
    parameters.allgather = allgather;
    parameters.req_test = request_test;
    parameters.req_free = request_free;
    parameters.data = (void*)(MPI_COMM_WORLD);
    parameters.rank = rank;
    parameters.nranks = size;
    parameters.local_device = localDeviceID;

    CAL_CHECK(cal_comm_create(parameters, &calCommunicator));

    return calCommunicator;
}
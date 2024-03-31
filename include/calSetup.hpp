#ifndef CAL_SETUP_HPP
#define CAL_SETUP_HPP

#include <cal.h>
#include <mpi.h>
#include <stdbool.h>
#include <string.h>
#include "errorHandling.hpp"
#include <unistd.h>
#include <cstdio>
#include <stdlib.h>

static calError_t allgather(void* src_buf, void* recv_buf, size_t size, void* data, void** request);
static calError_t request_test(void* request);
static calError_t request_free(void* request);
cal_comm_t createCalCommunicator(int rank, int size, int localDeviceID);

#endif
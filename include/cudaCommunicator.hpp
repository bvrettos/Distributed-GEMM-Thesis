#ifndef CUDA_COMMUNINCATOR_HPP
#define CUDA_COMMUNICATOR_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <stdio.h>
#include "errorHandling.hpp"

int getMaxGPUs();
void getLocalDevice(int* localRank, int* deviceCount, int* localDeviceID);
MPI_Comm createGPUCommunicator();

#endif
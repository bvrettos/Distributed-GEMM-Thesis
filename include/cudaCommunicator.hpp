#ifndef CUDA_COMMUNICATOR_HPP
#define CUDA_COMMUNICATOR_HPP

#include <cuda.h>
#include <cuda_runtime.h>
#include <mpi.h>
#include <cstdio>
#include <iostream>
#include <cmath>
#include "errorHandling.hpp"

#include <mpi-ext.h>

int getMaxGPUs();
void getLocalDevice(int* localRank, int* deviceCount, int* localDeviceID);
void calculateProcessGrid(int *dRow, int *dCol, int deviceCount);
MPI_Comm createGPUCommunicator();
bool checkCudaAwareMPI();

#endif
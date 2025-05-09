#include "cudaCommunicator.hpp"

void getLocalDevice(int* localRank, int* deviceCount, int* localDeviceID)
{
    MPI_Comm localCommunicator;

    /* Create a communicator per node */
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localCommunicator);
    MPI_Comm_rank(localCommunicator, localRank);
    MPI_Comm_free(&localCommunicator);

    /* Get number of devices on node */
    CUDA_CHECK(cudaGetDeviceCount(deviceCount));

    *localDeviceID = *localRank % *deviceCount;
    
    return;
}

int getMaxGPUs()
{
    MPI_Comm localCommunicator;

    /* Split into local nodes communicator */
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localCommunicator);

    /* Get localrank of each Node */
    int localRank;
    MPI_Comm_rank(localCommunicator, &localRank);

    /* Get number of GPUs per node */
    int localDevices = 0;
    if (localRank == 0) {
        CUDA_CHECK(cudaGetDeviceCount(&localDevices));
    }

    /* Reduce this number. Allreduce so that everyone gets the max amount */
    int allDevices = 0;
    MPI_Allreduce(&localDevices, &allDevices, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Comm_free(&localCommunicator);

    return allDevices;
}

void createGPUCommunicator(MPI_Comm* gpuCommunicator)
{
    MPI_Comm localCommunicator;
    int localSize, localRank, localDeviceSize, localDeviceRank;

    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &localCommunicator);
    MPI_Comm_rank(localCommunicator, &localRank);
    MPI_Comm_size(localCommunicator, &localSize);

    MPI_Group localGroup;
    MPI_Comm_group(localCommunicator, &localGroup);

    CUDA_CHECK(cudaGetDeviceCount(&localDeviceSize));

    /* Need to handle if numProcs is less than GPU size (meaning some GPUs are handled by same process - or remove them completely) */

    int deviceGroupRanks[localDeviceSize];
    for (int i = 0; i < localDeviceSize; i++) {
        deviceGroupRanks[i] = i;
    }

    MPI_Group deviceGroup;
    MPI_Group_incl(localGroup, localDeviceSize, deviceGroupRanks, &deviceGroup);

    MPI_Comm deviceCommunicator;
    MPI_Comm_create_group(localCommunicator, deviceGroup, 0, &deviceCommunicator);

    #ifdef DEBUG
        if (deviceCommunicator != MPI_COMM_NULL) {
            MPI_Comm_rank(deviceCommunicator, &localDeviceRank);
            MPI_Comm_size(deviceCommunicator, &localDeviceSizeComm);
            printf("Rank: %d, Size: %d\n", localDeviceRank, localDeviceSize);
        }
    #endif

    MPI_Comm_free(&localCommunicator);  

    *gpuCommunicator = deviceCommunicator;

    return;
}

bool checkCudaAwareMPI()
{
    return(MPIX_Query_cuda_support() == 1);
}
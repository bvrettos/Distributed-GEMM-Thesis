#include <generalUtilities.hpp>

int getSlurmNumNodes()
{
    return atoi(getenv("SLURM_JOB_NUM_NODES"));
}

double calculateGflops(const long long M, const long long N, const long long K, const double executionTime)
{
    return (2 * M * N * K * 1e-9) / executionTime;
}

void getGPUMemoryInfo(long long *freeMemory, long long *maxMemory, const int deviceID)
{
    /* Keep old device in order to return control to it */
    int previousDev=-2;
    CUDA_CHECK(cudaGetDevice(&previousDev));

    size_t freeMem, maxMem;
    CUDA_CHECK(cudaSetDevice(deviceID));
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &maxMem));

    *freeMemory = (long long) freeMem;
    *maxMemory = (long long) maxMem;

    #ifdef DEBUG
        std::cout << "GPUs MAX Vram (in GBs): " << maxMemory/(1024*1024*1024) << std::endl;
        std::cout << "GPUs Current free Vram (in GBs): " << freeMemory/(1024*1024*1024) << std::endl;
    #endif

    /* Return control to original device */
    CUDA_CHECK(cudaSetDevice(previousDev));

    return;
}

double csecond(void) {
  struct timespec tms;

  if (clock_gettime(CLOCK_REALTIME, &tms)) {
    return (0.0);
  }
  /// seconds, multiplied with 1 million
  int64_t micros = tms.tv_sec * 1000000;
  /// Add full microseconds
  micros += tms.tv_nsec / 1000;
  /// round up if necessary
  if (tms.tv_nsec % 1000 >= 500) {
    ++micros;
  }
  
  return ((double)micros / 1000000.0);
}
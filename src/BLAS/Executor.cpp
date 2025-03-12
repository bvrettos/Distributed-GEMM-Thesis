#include <Executor.hpp>

#define MAX_PARALLELISM_FACTOR 8


int calculateMaxParallelizationFactor(int dRow, int dCol, int dStack, 
    int blockRows, int blockColumns, int m, int n, int k, size_t sizeOfElement)
{
    /* Algorithm's Parallelization Limit */
    int min2D = std::min(dRow, dCol);
    int min25D = dStack* min2D;
    int numberOfSteps = static_cast<int>(std::ceil(k/blockColumns))/dStack; // Number of Steps (cannot be higher than that)

    /* Memory Limit for Parallelization */
    long long freeMemory, maxMemory;
    getGPUMemoryInfo(&freeMemory, &maxMemory, 0);
    long long memoryRequirementsPerTask = 3*blockRows*blockColumns*sizeOfElement;
    int memoryFactor = freeMemory/memoryRequirementsPerTask;
    
    /* In total there are 3 limits, the actual algorithm's limit, the memory limit and the hard-cap limit that we have set. */
    return std::min({min25D, memoryFactor, MAX_PARALLELISM_FACTOR, numberOfSteps});
}

void allocateWorkspaceMemory()
{
    /* Calculate max parallelization factor */

    /* Streams should have already been created, just create the workspace for execution */
}
#ifndef EXECUTOR_HPP
#define EXECUTOR_HPP

#include <cmath>
#include <generalUtilities.hpp>
#include <cstdio>
#include <cuda.h>
#include <cublas_v2.h> 

class Executor {
    private:
        /* GPU Calculations */
        cublasHandle_t cublasContext;
        cudaStream_t* communicationStreamPool;
        cudaStream_t* computationStreamPool;

        /* CPU Calculations */
        // TODO??: Implement a small parallel CPU execution system
    public:

};


/* Create Tasks as classes, attach functions for them to call
    - We can create L1, L2 and L3 tasks with inheritance 
    - Use function pointers or C++ lambdas to call them
    - Synchronize within Executor (CUDA Streams for GPUs and locks for CPUs)
*/

class BlasTask {
    private:

    public:
}




#endif
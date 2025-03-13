#include <MatrixUtilities.hpp>

template <typename scalar_t>
void printMatrix(scalar_t* array, long long rows, long long columns, int rank, MatrixLayout layout)
{
    assert(array != nullptr);

    printf("Rank: %d\n", rank);
    if (layout == MatrixLayout::ColumnMajor)
        std::swap(rows, columns);
    
    for (long long i = 0; i < rows; i++) {
        for (long long j = 0; j < columns; j++) {
            printf("%lf ", array[i*columns + j]);;
        }
        printf("\n");
    }
    printf("\n");

    return;
}

template <typename scalar_t>
void writeMatrix(scalar_t* array, long long rows, long long columns, std::string& filename, MatrixLayout layout)
{
    std::ofstream outFile(filename, std::ofstream::out | std::ofstream::app);

    if (!outFile.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
    }

    if (layout == MatrixLayout::ColumnMajor)
        std::swap(rows, columns);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            outFile << array[i*columns + j] << " ";
        }
        outFile << std::endl;
    }
    outFile << std::endl;
    outFile.close();

    return;
}

template <typename scalar_t>
void generateMatrix(scalar_t* array, const long long size)
{
    struct timespec dummy;
    clock_gettime(CLOCK_REALTIME, &dummy);
    srand(dummy.tv_nsec);
    for (long long i = 0; i < size; i++) 
        array[i] = (scalar_t) rand() / rand();

    return;
}

template <typename scalar_t>
void generateMatrix(scalar_t* array, const long long rows, const long long columns)
{
    long long size = rows*columns;
    generateMatrix(array, size);

    return;
}

template <typename scalar_t>
void generateMatrixGPU(scalar_t* array, const long long size, const int deviceID)
{
    int prevDeviceID = -2;
    CUDA_CHECK(cudaGetDevice(&prevDeviceID));

    CUDA_CHECK(cudaSetDevice(deviceID));

    long long freeMemory, maxMemory;
    getGPUMemoryInfo(&freeMemory, &maxMemory, deviceID);

    long long bytesToGenerate = size * sizeof(scalar_t);

    if (bytesToGenerate > freeMemory) {
        /* Find out how many times to divide the matrix (row-wise). */
        int partitions = (bytesToGenerate/freeMemory) + 1;

        /* Cut down rows in equal partitions */
        for (int i = 0; i < partitions; i++) {
            long long elementsToGenerate = size/partitions;
            long long offset = elementsToGenerate*i;

            if ((i+1) == partitions) elementsToGenerate += size%partitions;
            generateMatrixGPU(&array[offset], elementsToGenerate, deviceID);
        }
        return;
    }

    // Base Case: Size fits GPU memory
    curandGenerator_t generator;

    /* Check if pointer is device or host memory */
    cudaPointerAttributes pointerAttributes;
    CUDA_CHECK(cudaPointerGetAttributes(&pointerAttributes, array));

    /* If memory is already allocated in device, you can simply generate data there */
    if (pointerAttributes.type == cudaMemoryTypeDevice) {
        curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
        struct timespec dummy;
        clock_gettime(CLOCK_REALTIME, &dummy);
        curandSetPseudoRandomGeneratorSeed(generator, dummy.tv_nsec);
        
        if (typeid(scalar_t) ==  typeid(float)) curandGenerateUniform(generator, (float*) array, size);
        else if (typeid(scalar_t) ==  typeid(double)) curandGenerateUniformDouble(generator, (double*) array, size);
    }

    /* If not, then generate data on GPU and then copy it back to host */
    else if (pointerAttributes.type == cudaMemoryTypeHost) {
        scalar_t* matrix; // GPU Matrix

        CUDA_CHECK(cudaMalloc((void **)&matrix, bytesToGenerate));
        curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
        struct timespec dummy;
        clock_gettime(CLOCK_REALTIME, &dummy);
        curandSetPseudoRandomGeneratorSeed(generator, dummy.tv_nsec);
        
        if (typeid(scalar_t) ==  typeid(float)) curandGenerateUniform(generator, (float*) matrix, size);
        else if (typeid(scalar_t) ==  typeid(double)) curandGenerateUniformDouble(generator, (double*) matrix, size);

        CUDA_CHECK(cudaMemcpy(array, matrix, size, cudaMemcpyDeviceToHost)); // Copy generated matrix to host memory

        CUDA_CHECK(cudaFree(matrix)); // De-allocate GPU matrix
    }

    return;
}

template <typename scalar_t>
void generateMatrixGPU(scalar_t* array, const long long rows, const long long columns, const int deviceID)
{
    long long size = rows * columns;
    generateMatrixGPU(array, size, deviceID);
}

// Strict Instantiations
template void printMatrix<float>(float* array, long long rows, long long columns, int rank, MatrixLayout layout);
template void printMatrix<double>(double* array, long long rows, long long columns, int rank, MatrixLayout layout);

template void writeMatrix<float>(float* array, long long rows, long long columns, std::string& filename, MatrixLayout layout);
template void writeMatrix<double>(double* array, long long rows, long long columns, std::string& filename, MatrixLayout layout);

template void generateMatrix<float>(float *array, const long long rows, const long long columns);
template void generateMatrix<double>(double *array, const long long rows, const long long columns);

template void generateMatrix<float>(float *array, const long long size);
template void generateMatrix<double>(double *array, const long long size);

template void generateMatrixGPU<double>(double* array, const long long rows, const long long columns, const int deviceID);
template void generateMatrixGPU<float>(float* array, const long long rows, const long long columns, const int deviceID);

template void generateMatrixGPU<double>(double* array, const long long size, const int deviceID);
template void generateMatrixGPU<float>(float* array, const long long size, const int deviceID);
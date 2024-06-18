#include "cmatrix.h"

void printMatrix(double *array, const long long rows, const long long columns, int rank) {
    std::cout << "RANK: " << rank << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            std::cout << array[i*columns + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void printMatrixColumnMajor(double *array, const long long rows, const long long columns, const int rank)
{
    printf("RANK: %d\n", rank);
    for (int i = 0; i < rows; i++) {
        for (int j = 0 ; j < columns; j++) {
            printf("%lf ", array[j*rows + i]);
        }
        printf("\n");
    }
}

void printInLine(double *array, int rows, int columns, int rank) {
    std::cout << "RANK: " << rank << std::endl;
    for (int i = 0; i < rows*columns; i++)
        std::cout << array[i] << " ";
    std::cout << std::endl;
}

void writeMatrixToFile(double* array, const int rows, const int columns, const std::string& filename)
{
    std::ofstream outFile(filename, std::ofstream::out | std::ofstream::app);

    if (!outFile.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
    }
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            outFile << array[i*columns + j] << " ";
        }
        outFile << std::endl;
    }
    outFile << std::endl;
    outFile.close();
}


void writeMatrixToFileColumnMajor(double* array, const int rows, const int columns, const std::string& filename)
{
    std::ofstream outFile(filename, std::ofstream::out | std::ofstream::app);

    if (!outFile.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            outFile << array[j*rows + i] << " ";
        }
        outFile << std::endl;
    }
    outFile << std::endl;

    outFile.close();
}

/* Generation Functions*/
template <typename T>
void generateMatrix(T *array, const long long rows, const long long columns) {
    struct timespec dummy;
    clock_gettime(CLOCK_REALTIME, &dummy);
    srand(dummy.tv_nsec);
    for (long long i = 0; i < rows; i++) {
        for (long long j = 0; j < columns; j++) {
            array[i*columns + j] = (T) rand() / rand();
        }
    }
}

/* Generation Functions*/
template <typename T>
void generateMatrix(T *array, const long long size) {
    struct timespec dummy;
    clock_gettime(CLOCK_REALTIME, &dummy);
    srand(dummy.tv_nsec);
    for (long long i = 0; i < size; i++) 
        array[i] = (T) rand() / rand();
}


template <typename T>
T* copyMatrix(T *matrix, const long long rows, const long long columns) {
    T *copiedMatrix = (T *) malloc(sizeof(T)*rows*columns);
    for (long long i = 0; i < rows*columns; i++) {
        copiedMatrix[i] = matrix[i];
    }
    return copiedMatrix;
}

template <typename T>
void generateMatrixGPU(T* array, const long long size, const int deviceID)
{
    int prevDeviceID = -2;
    CUDA_CHECK(cudaGetDevice(&prevDeviceID));

    CUDA_CHECK(cudaSetDevice(deviceID));

    long long freeMemory, maxMemory;
    getGPUMemoryInfo(&freeMemory, &maxMemory, deviceID);

    long long bytesToGenerate = size * sizeof(T);

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

    T* matrix; // GPU Matrix

    CUDA_CHECK(cudaMalloc((void **)&matrix, bytesToGenerate));
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    struct timespec dummy;
    clock_gettime(CLOCK_REALTIME, &dummy);
    curandSetPseudoRandomGeneratorSeed(generator, dummy.tv_nsec);
    
    if (typeid(T) ==  typeid(float)) curandGenerateUniform(generator, (float*) matrix, size);
    else if (typeid(T) ==  typeid(double)) curandGenerateUniformDouble(generator, (double*) matrix, size);

    CUDA_CHECK(cudaMemcpy(array, matrix, size, cudaMemcpyDeviceToHost)); // Copy generated matrix to host memory

    CUDA_CHECK(cudaFree(matrix)); // De-allocate GPU matrix

    return;
}

template <typename T>
void generateMatrixGPU(T* array, const long long rows, const long long columns, const int deviceID)
{
    CUDA_CHECK(cudaSetDevice(deviceID));

    /* Get max GPU Size */
    cudaDeviceProp deviceProperties;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, deviceID));
    long long freeMemory, maxMemory;
    getGPUMemoryInfo(&freeMemory, &maxMemory, deviceID);

    /* Cannot generate more that these bytes */
    long long bytesToGenerate = sizeof(T) * rows * columns;

    if (bytesToGenerate > freeMemory) {
        /* Find out how many times to divide the matrix (row-wise). */
        int partitions = (bytesToGenerate/freeMemory) + 1;

        /* Cut down rows in equal partitions */
        for (int i = 0; i < partitions; i++) {
            long long rowsToGenerate = rows/partitions;
            if ((i+1) == partitions) rowsToGenerate += rows%partitions;
            generateMatrixGPU(&array[rows*columns*i/partitions], rowsToGenerate, columns, deviceID);
        }
    
        // return;
    }

    long long int size = rows*columns;
    curandGenerator_t generator;

    T* matrix; // GPU Matrix

    CUDA_CHECK(cudaMalloc((void **)&matrix, rows * columns * sizeof(T)));
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    struct timespec dummy;
    clock_gettime(CLOCK_REALTIME, &dummy);
    curandSetPseudoRandomGeneratorSeed(generator, dummy.tv_nsec);
    
    if (typeid(T) ==  typeid(float)) curandGenerateUniform(generator, (float*) matrix, size);
    else if (typeid(T) ==  typeid(double)) curandGenerateUniformDouble(generator, (double*) matrix, size);

    CUDA_CHECK(cudaMemcpy(array, matrix, rows*columns*sizeof(T), cudaMemcpyDeviceToHost)); // Copy generated matrix to host memory

    CUDA_CHECK(cudaFree(matrix)); // De-allocate GPU matrix
    return;
}

template <typename T>
void MatrixInit(T *matrix, const long long rows, const long long columns, int loc)
{
    if (!matrix) {
        printf("Matrix not initialized in the right way\n");
        exit(-1);
    }
    /* Get size of GPUs */
    int maxGPUs = 0;
    CUDA_CHECK(cudaGetDeviceCount(&maxGPUs));

    if (loc < -2 || loc >= maxGPUs)
    {
        printf("Location == -1 (CPU), Location {0,...gpuSize-1} == DeviceID \n");
        exit(-2);
    }
    else if (loc == -1) {
        printf("Generating Matrix: M=%lld, N=%lld on CPU\n", rows, columns);
        generateMatrix(matrix, rows, columns); // CPU generation (slow but much more memory)
    }
    
    else {
        printf("Generating Matrix: M=%lld, N=%lld on GPU with devId=%d\n", rows, columns, loc);
        int prevLoc;
        cudaGetDevice(&prevLoc);
        generateMatrixGPU(matrix, rows, columns, loc); // GPU generation
        cudaSetDevice(prevLoc);
    }
}

template <typename T>
void MatrixInit(T *matrix, const long long size, int loc)
{
    if (!matrix) {
        printf("Matrix not initialized in the right way\n");
        exit(-1);
    }
    /* Get size of GPUs */
    int maxGPUs = 0;
    CUDA_CHECK(cudaGetDeviceCount(&maxGPUs));

    if (loc < -2 || loc >= maxGPUs)
    {
        printf("Location == -1 (CPU), Location {0,...gpuSize-1} == DeviceID \n");
        exit(-2);
    }
    else if (loc == -1) {
        printf("Generating Matrix: Size=%lld on CPU\n", size);
        generateMatrix(matrix, size); // CPU generation (slow but much more memory)
    }
    
    else {
        printf("Generating Matrix: Size=%lld on GPU with devId=%d\n", size, loc);
        int prevLoc;
        cudaGetDevice(&prevLoc);
        generateMatrixGPU(matrix, size, loc); // GPU generation
        cudaSetDevice(prevLoc);
    }
}

template void generateMatrixGPU<double>(double* array, const long long rows, const long long columns, const int deviceID);
template void generateMatrixGPU<float>(float* array, const long long rows, const long long columns, const int deviceID);
template void generateMatrixGPU<double>(double* array, const long long size, const int deviceID);
template void generateMatrixGPU<float>(float* array, const long long size, const int deviceID);

template void MatrixInit<double>(double* matrix, const long long rows, const long long columns, int loc);
template void MatrixInit<float>(float* matrix, const long long rows, const long long columns, int loc);
template void MatrixInit<double>(double* matrix, const long long size, int loc);
template void MatrixInit<float>(float* matrix, const long long size, int loc);

template void generateMatrix<double>(double* array, const long long rows, const long long columns);
template void generateMatrix<float>(float* array, const long long rows, const long long columns);
template void generateMatrix<double>(double* array, const long long size);
template void generateMatrix<float>(float* array, const long long size);

template double* copyMatrix<double>(double* matrix, const long long rows, const long long columns);
template float* copyMatrix<float>(float* matrix, const long long rows, const long long columns);
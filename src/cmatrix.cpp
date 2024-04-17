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

void printMatrixColumnMajor(double *array, const int rows, const int columns, const int rank)
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
    srand(rows*columns);
    for (long long i = 0; i < rows; i++) {
        for (long long j = 0; j < columns; j++) {
            array[i*columns + j] = (T) rand() / rand();
        }
    }
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
void generateMatrixGPU(T* array, const long long rows, const long long columns, const int deviceID)
{
    cudaSetDevice(deviceID);

    long long int size = rows*columns;
    curandGenerator_t generator;

    T* matrix; // GPU Matrix

    CUDA_CHECK(cudaMalloc((void **)&matrix, rows * columns * sizeof(T)));
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, 3878593);
    
    if (typeid(T) ==  typeid(float)) curandGenerateUniform(generator, (float*) matrix, size);
    else if (typeid(T) ==  typeid(double)) curandGenerateUniformDouble(generator, (double*) matrix, size);

    CUDA_CHECK(cudaMemcpy(array, matrix, rows*columns*sizeof(T), cudaMemcpyDeviceToHost)); // Copy generated matrix to host memory

    CUDA_CHECK(cudaFree(matrix)); // De-allocate GPU matrix
}

template <typename T>
void MatrixInit(T *matrix, const long long rows, const long long columns, int loc)
{
    if (!matrix) {
        printf("Matrix not initialized in the right way\n");
        exit(-1);
    }
     /* Get length of GPUs */
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
        int prevLoc;
        cudaGetDevice(&prevLoc);
        generateMatrixGPU(matrix, rows, columns, loc); // GPU generation (fast but about 16k x 16k can fit);
        cudaSetDevice(prevLoc);
    }
}

template <typename T>
void generateGemmMatrices(T* A, T* B, T* C, const long long M, const long long N, const long long K, int loc)
{
    A = (T*) malloc(sizeof(T) * M * K);
    B = (T*) malloc(sizeof(T) * N * K);
    C = (T*) malloc(sizeof(T) * M * N);

    MatrixInit(A, M, K, loc);
    MatrixInit(B, K, N, loc);
    MatrixInit(C, M, N, loc);

    return;
}

template void generateMatrixGPU<double>(double* array, const long long rows, const long long columns, const int deviceID);
template void generateMatrixGPU<float>(float* array, const long long rows, const long long columns, const int deviceID);

template void MatrixInit<double>(double* matrix, const long long rows, const long long columns, int loc);
template void MatrixInit<float>(float* matrix, const long long rows, const long long columns, int loc);

template void generateGemmMatrices<double>(double* A, double *B, double* C, const long long M, const long long N, const long long K, int loc);
template void generateGemmMatrices<float>(float* A, float *B, float* C, const long long M, const long long N, const long long K, int loc);

template void generateMatrix<double>(double* array, const long long rows, const long long columns);
template void generateMatrix<float>(float* array, const long long rows, const long long columns);

template double* copyMatrix<double>(double* matrix, const long long rows, const long long columns);
template float* copyMatrix<float>(float* matrix, const long long rows, const long long columns);
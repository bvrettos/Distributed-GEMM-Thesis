#include <DistributedMatrix.hpp>
#include <cxxopts.hpp>
#include "cudaCommunicator.hpp"

int main(int argc, char* argv[])
{

    cxxopts::Options options("DistributedMatrix SUMMA Validator", "Validator program");
    options.add_options()
        ("m,m_dim",
            "number of rows of A and C.", 
            cxxopts::value<int64_t>()->default_value("4096"))
        ("n,n_dim",
            "number of columns of B and C.",
            cxxopts::value<int64_t>()->default_value("4096"))
        ("k,k_dim",
            "number of columns of A and rows of B.", 
            cxxopts::value<int64_t>()->default_value("4096"))
        ("mb",
            "rows blocking dimension", 
            cxxopts::value<int64_t>()->default_value("512"))
        ("nb",
            "columns blocking dimension", 
            cxxopts::value<int64_t>()->default_value("512"))
        ("h,help", "Print usage.")
        ;

    auto result = options.parse(argc, argv);

    long long m = result["m_dim"].as<int64_t>();
    long long n = result["n_dim"].as<int64_t>();
    long long k = result["k_dim"].as<int64_t>();
    long long mb = result["mb"].as<int64_t>();
    long long nb = result["nb"].as<int64_t>();

    MPI_Init(&argc, &argv);
    int cpuRank, cpuSize, gpuRank, gpuSize;

    /* Create GPU Communicator - Each device gets it's own process */
    MPI_Comm gpuCommunicator;
    createGPUCommunicator(&gpuCommunicator);
    MPI_Comm_rank(gpuCommunicator, &gpuRank);
    MPI_Comm_size(gpuCommunicator, &gpuSize);
    printf("Rank: %d\n", gpuRank);
    CUDA_CHECK(cudaSetDevice(gpuRank));

    if (gpuRank == 0) printf("Rank: %d (M=%lld, N=%lld, K=%lld) - Number of Devices: %d\n", gpuRank, m, n, k, gpuSize);

    DistributedMatrix<double> matrixA = generateRandomMatrix<double>(m, k, mb, nb, MemoryLocation::Device, gpuCommunicator, MatrixLayout::ColumnMajor);

    matrixA.print();
    if (matrixA.tileIsMine(0, 0)) {
        matrixA.getTile(1, 0).printTile(gpuRank);
    }
    
    

    MPI_Finalize();
    return 0;
}
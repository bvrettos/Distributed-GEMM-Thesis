#include <slateWrapped.hpp>

template <typename matrix_type>
void random_matrix(matrix_type& A)
{
    for (int64_t j = 0; j < A.nt(); ++j) {
        for (int64_t i = 0; i < A.mt(); ++i) {
            if (A.tileIsLocal( i, j )) {
                try {
                    auto T = A( i, j );
                    MatrixInit(T.data(), T.mb(), T.nb());
                    // random_matrix( T.mb(), T.nb(), T.data(), T.stride() );
                }
                catch (...) {
                    // ignore missing tiles
                }
            }
        }
    }
}

template <typename scalar_type>
void slateFullGemmOffload(char TransA,  char TransB, const long long M, const long long N, const long long K,
  scalar_type alpha, scalar_type* A, long long int ldA, scalar_type* B, long long int ldB, scalar_type beta, scalar_type* C, long long int ldC
  ,long mb, long nb, bool logging, bool gatherResults, int initialDataLocation)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dRow, dCol; 
    calculateProcessGrid(&dRow, &dCol, size);
    int numberOfNodes = getSlurmNumNodes();
    int totalGPUs = size;
    int devId = 0;

    /* Get memory size to calculate lookahead later. Get free memory size before inserting tiles into device */
    long long freeMemory, maxMemory;
    getGPUMemoryInfo(&freeMemory, &maxMemory, devId); //device id is = 0 because each rank receives its own device

    /* By default, insert tiles to CPU (meaning all host) */
    slate::Target initialTileMemoryLocation = (initialDataLocation == 0) ? slate::Target::Devices : slate::Target::Host;
    std::string dataLocation = (initialDataLocation == 0) ? "devices" : "host"; 

    FILE* logfile;
    if (logging) {
        if (rank == 0) {
            std::string machineName = MACHINE_NAME;
            std::string filename = "DGEMM_execution_logs-Full_GEMM_Offload-" + machineName + "-SLATE.csv";
            std::string header = "Algo,M,N,K,TileRows,TileColumns,dRow,dCol,TotalNodes,TotalGPUs,DecompositionTime,ExecutionTime,GatherTime,GFlops,DataLocation";
            logfile = createLogCsv(filename, header);
        }
    }

    double decomposeBefore = MPI_Wtime();
    /* Change this to PBLAS Decomposition */
    slate::Matrix<scalar_type> matrixA = slate::Matrix<scalar_type>::fromLAPACK(M, K, A, ldA, mb, nb, dRow, dCol, MPI_COMM_WORLD);
    slate::Matrix<scalar_type> matrixB = slate::Matrix<scalar_type>::fromLAPACK(K, N, B, ldB, mb, nb, dRow, dCol, MPI_COMM_WORLD);
    slate::Matrix<scalar_type> matrixC = slate::Matrix<scalar_type>::fromLAPACK(M, N, C, ldC, mb, nb, dRow, dCol, MPI_COMM_WORLD);

    matrixA.insertLocalTiles(initialTileMemoryLocation);
    matrixB.insertLocalTiles(initialTileMemoryLocation);
    matrixC.insertLocalTiles(initialTileMemoryLocation);
    double decomposeAfter = MPI_Wtime();

    double executionBefore, executionAfter;

    int totalDeviceTiles = 0;
    totalDeviceTiles += matrixA.getMaxDeviceTiles(0);
    totalDeviceTiles += matrixB.getMaxDeviceTiles(0);
    totalDeviceTiles += matrixC.getMaxDeviceTiles(0);

    long long tileMemorySize = mb * nb;
    if (typeid(scalar_type) == typeid(float)) tileMemorySize *= sizeof(float);
    else if (typeid(scalar_type) == typeid(double)) tileMemorySize *= sizeof(double);
    int lookahead = std::min((long long) totalDeviceTiles, freeMemory/tileMemorySize);
    
    printf("Rank %d has %d total tiles. Lookahead is %d\n", rank, totalDeviceTiles, lookahead);

    /* Execute on GPU */
    if (blas::get_device_count() > 0) {
        slate::Options opts = {
            { slate::Option::Lookahead, lookahead},
            { slate::Option::Target, slate::Target::Devices},
        };

        /* I think that there should be barriers here (for correct times) */
        MPI_Barrier(MPI_COMM_WORLD);
        executionBefore = MPI_Wtime();
        slate::gemm(alpha, matrixA, matrixB, beta, matrixC, opts);
        MPI_Barrier(MPI_COMM_WORLD);
        executionAfter = MPI_Wtime();
    }
    double gatherBefore=0, gatherAfter=0;
    /* Result is gathered in rank = 0 */
    if (gatherResults) {
        gatherBefore = MPI_Wtime();
        matrixC.gather(C, ldC);
        gatherAfter = MPI_Wtime();
    }

    if (logging) {
        if (rank == 0) {
            double executionTime = executionAfter - executionBefore;
            double decompositionTime = decomposeAfter - decomposeBefore;
            double gatherTime = gatherAfter - gatherBefore;
            double gflops = calculateGflops(M, N, K, executionTime);

            char csvLine[200];
            sprintf(csvLine, "%s,%lld,%lld,%lld,%lld,%lld,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%s\n",
                            "SLATE", M, N, K, mb, nb, dRow, dCol, numberOfNodes, totalGPUs, 
                            decompositionTime, executionTime, gatherTime, gflops, dataLocation);
            writeLineToFile(logfile, csvLine);
        }
    }

    return;
}

template <typename scalar_type>
void slatePreDistributedGemm(char TransA,  char TransB, const long long M, const long long N, const long long K,
  scalar_type alpha, scalar_type* A, long long int ldA, scalar_type* B, long long int ldB, scalar_type beta, scalar_type* C, long long int ldC
  ,long mb, long nb, int numberOfRuns, bool logging, bool gatherResults, int initialDataLocation)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dRow, dCol; 
    calculateProcessGrid(&dRow, &dCol, size);
    int numberOfNodes = getSlurmNumNodes();
    int totalGPUs = size;

    int devId = 0;
    /* Get memory size to calculate lookahead later. Get free memory size before inserting tiles into device */
    long long freeMemory, maxMemory;
    getGPUMemoryInfo(&freeMemory, &maxMemory, devId); //device id is = 0 because each rank receives its own device

    /* By default, insert tiles to CPU (meaning all host) */
    slate::Target initialTileMemoryLocation = (initialDataLocation == 0) ? slate::Target::Devices : slate::Target::Host;
    std::string dataLocation = (initialDataLocation == 0) ? "devices" : "host"; 
    FILE* logfile;

    if (logging) {    
        if (rank == 0) {
            std::string machineName = MACHINE_NAME;
            std::string filename = "DGEMM_execution_logs-PreDistributed_GEMM-" + machineName + "-SLATE.csv";
            std::string header = "Algo,M,N,K,TileRows,TileColumns,dRow,dCol,TotalNodes,TotalGPUs,DecompositionTime,ExecutionTime,GatherTime,GFlops,DataLocation";
            logfile = createLogCsv(filename, header);
        }
    }

    double decomposeBefore = MPI_Wtime();
    slate::Matrix<scalar_type> matrixA(M, K, mb, nb, dRow, dCol, MPI_COMM_WORLD);
    slate::Matrix<scalar_type> matrixB(K, N, mb, nb, dRow, dCol, MPI_COMM_WORLD);
    slate::Matrix<scalar_type> matrixC(M, N, mb, nb, dRow, dCol, MPI_COMM_WORLD);
    
    random_matrix(matrixA);
    random_matrix(matrixB);
    random_matrix(matrixC);

    matrixA.insertLocalTiles(initialTileMemoryLocation);
    matrixB.insertLocalTiles(initialTileMemoryLocation);
    matrixC.insertLocalTiles(initialTileMemoryLocation);
    double decomposeAfter = MPI_Wtime();

    int totalDeviceTiles = 0;
    totalDeviceTiles += matrixA.getMaxDeviceTiles(0);
    totalDeviceTiles += matrixB.getMaxDeviceTiles(0);
    totalDeviceTiles += matrixC.getMaxDeviceTiles(0);

    long long tileMemorySize = mb * nb;
    if (typeid(scalar_type) == typeid(float)) tileMemorySize *= sizeof(float);
    else if (typeid(scalar_type) == typeid(double)) tileMemorySize *= sizeof(double);
    int lookahead = std::min((long long) totalDeviceTiles, freeMemory/tileMemorySize);

    printf("Rank %d has %d total tiles. Lookahead is %d\n", rank, totalDeviceTiles, lookahead);

    double executionBefore, executionAfter;

    /* Options to execute on GPU */
    slate::Options opts = {
        { slate::Option::Lookahead, lookahead},
        { slate::Option::Target, slate::Target::Devices},
    };

    for (int i = 0; i < numberOfRuns; i++) {
        /* I think that there should be barriers here (for correct times) */
        MPI_Barrier(MPI_COMM_WORLD);
        executionBefore = MPI_Wtime();
        slate::gemm(alpha, matrixA, matrixB, beta, matrixC, opts);
        MPI_Barrier(MPI_COMM_WORLD);
        executionAfter = MPI_Wtime();

        double gatherBefore=0, gatherAfter=0;
        /* Result is gathered in rank = 0 */
        if (gatherResults) {
            gatherBefore = MPI_Wtime();
            matrixC.gather(C, ldC);
            gatherAfter = MPI_Wtime();
        }

        if (logging) {
            if (rank == 0) {
                double executionTime = executionAfter - executionBefore;
                double decompositionTime = decomposeAfter - decomposeBefore;
                double gatherTime = gatherAfter - gatherBefore;
                double gflops = calculateGflops(M, N, K, executionTime);

                char csvLine[250];
                sprintf(csvLine, "%s,%lld,%lld,%lld,%lld,%lld,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%s\n",
                                "SLATE", M, N, K, mb, nb, dRow, dCol, numberOfNodes, totalGPUs, 
                                decompositionTime, executionTime, gatherTime, gflops, dataLocation.c_str());
                writeLineToFile(logfile, csvLine);
            }
        }
    }

    return;
}

template <typename scalar_type>
void validateGEMM(char TransA,  char TransB, const long long M, const long long N, const long long K,
  scalar_type alpha, scalar_type* A, long long int ldA, scalar_type* B, long long int ldB, scalar_type beta, scalar_type* C, long long int ldC
  ,long mb, long nb)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* This runs on CPU so it is pretty slow :( */
    scalar_type* referenceC;
    if (rank == 0) {
        referenceC = copyMatrix(C, M, N);
    }
    
    /* Run SLATE GEMM */
    slateGEMM(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC, mb, nb, false, true, 0);

    if (rank == 0) {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, ldA, B, ldB, beta, referenceC, ldC);
        Dtest_equality(C, referenceC, M*N);
    }

    return;
}

template <typename scalar_type>
void slateGEMM(char TransA,  char TransB, const long long M, const long long N, const long long K,
  scalar_type alpha, scalar_type* A, long long int ldA, scalar_type* B, long long int ldB, scalar_type beta, scalar_type* C, long long int ldC
  ,long mb, long nb, bool logging, bool gatherResults, int initialDataLocation)
{    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dRow, dCol; 
    calculateProcessGrid(&dRow, &dCol, size);
    int numberOfNodes = getSlurmNumNodes();
    int totalGPUs = size;
    int lookahead = 2;

    /* By default, insert tiles to CPU (meaning all host) */
    slate::Target initialTileMemoryLocation = (initialDataLocation == 0) ? slate::Target::Devices : slate::Target::Host;
    std::string dataLocation = (initialDataLocation == 0) ? "devices" : "host"; 

    FILE* logfile;
    if (logging) {
        if (rank == 0) {
            std::string machineName = MACHINE_NAME;
            std::string filename = "DGEMM_execution_logs-" + machineName + "-SLATE.csv";
            std::string header = "Algo,M,N,K,TileRows,TileColumns,dRow,dCol,TotalNodes,TotalGPUs,DecompositionTime,ExecutionTime,GatherTime,GFlops,DataLocation";
            logfile = createLogCsv(filename, header);
        }
    }

    double decomposeBefore = MPI_Wtime();
        /*NOTE: This is very stupid. Talk to Petros about benchmarking by generating data in the tiles instead of copying whole Matrices between nodes */
        slate::Matrix<scalar_type> matrixA = slate::Matrix<scalar_type>::fromLAPACK(M, K, A, ldA, mb, nb, dRow, dCol, MPI_COMM_WORLD);
        slate::Matrix<scalar_type> matrixB = slate::Matrix<scalar_type>::fromLAPACK(K, N, B, ldB, mb, nb, dRow, dCol, MPI_COMM_WORLD);
        slate::Matrix<scalar_type> matrixC = slate::Matrix<scalar_type>::fromLAPACK(M, N, C, ldC, mb, nb, dRow, dCol, MPI_COMM_WORLD);

        matrixA.insertLocalTiles(initialTileMemoryLocation);
        matrixB.insertLocalTiles(initialTileMemoryLocation);
        matrixC.insertLocalTiles(initialTileMemoryLocation);
    double decomposeAfter = MPI_Wtime();

    double executionBefore, executionAfter;

    /* Execute on GPU */
    if (blas::get_device_count() > 0) {
        slate::Options opts = {
            { slate::Option::Lookahead, lookahead},
            { slate::Option::Target, slate::Target::Devices},
        };

        /* I think that there should be barriers here (for correct times) */
        MPI_Barrier(MPI_COMM_WORLD);
        executionBefore = MPI_Wtime();
        slate::gemm(alpha, matrixA, matrixB, beta, matrixC, opts);
        MPI_Barrier(MPI_COMM_WORLD);
        executionAfter = MPI_Wtime();
    }

    double gatherBefore=0, gatherAfter=0;
    /* Result is gathered in rank = 0 */
    if (gatherResults) {
        gatherBefore = MPI_Wtime();
        matrixC.gather(C, ldC);
        gatherAfter = MPI_Wtime();
    }

    if (logging) {
        if (rank == 0) {
            double executionTime = executionAfter - executionBefore;
            double decompositionTime = decomposeAfter - decomposeBefore;
            double gatherTime = gatherAfter - gatherBefore;
            double gflops = calculateGflops(M, N, K, executionTime);

            char csvLine[200];
            sprintf(csvLine, "%s,%lld,%lld,%lld,%lld,%lld,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%s\n",
                            "SLATE", M, N, K, mb, nb, dRow, dCol, numberOfNodes, totalGPUs, 
                            decompositionTime, executionTime, gatherTime, gflops, dataLocation.c_str());
            writeLineToFile(logfile, csvLine);
        }
    }

    return;
}

template void slateGEMM<double>(char TransA,  char TransB, const long long M, const long long N, const long long K,
  double alpha, double* A, long long int ldA, double* B, long long int ldB, double beta, double* C, long long int ldC, long mb, long nb, bool logging=false, bool gatherResults=false, int initialDataLocation=-1);
template void slateGEMM<float>(char TransA,  char TransB, const long long M, const long long N, const long long K,
  float alpha, float* A, long long int ldA, float* B, long long int ldB, float beta, float* C, long long int ldC, long mb, long nb, bool logging=false, bool gatherResults=false, int initialDataLocation=-1);

template void validateGEMM<double>(char TransA,  char TransB, const long long M, const long long N, const long long K,
  double alpha, double* A, long long int ldA, double* B, long long int ldB, double beta, double* C, long long int ldC
  ,long mb, long nb);

template void slatePreDistributedGemm<double>(char TransA,  char TransB, const long long M, const long long N, const long long K,
  double alpha, double* A, long long int ldA, double* B, long long int ldB, double beta, double* C, long long int ldC
  ,long mb, long nb, int numberOfRuns, bool logging, bool gatherResults, int initialDataLocation);

template void slateFullGemmOffload<double>(char TransA,  char TransB, const long long M, const long long N, const long long K,
  double alpha, double* A, long long int ldA, double* B, long long int ldB, double beta, double* C, long long int ldC
  ,long mb, long nb, bool logging, bool gatherResults, int initialDataLocation);
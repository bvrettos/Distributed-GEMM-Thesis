#include <cosmaWrapped.hpp>

template <typename scalar_t>
void cosmaFullGemmOffloadScalapack(char TransA, char TransB, const long long M, const long long N, const long long K, scalar_t alpha, scalar_t* A, const long long lda,
    scalar_t* B, const long long ldb, scalar_t beta, scalar_t* C, const long long ldc, int Mb, int Nb, MPI_Comm comm, const int numberOfRuns, bool logging, bool gatherResult)
{
    /* Use ScaLAPACK Layout and distribute A, B and C from root host to others. Use pdgemm wrapper of COSMA */
    using namespace cosma;
    using namespace costa;

    int initialized;
    MPI_Initialized(&initialized);

    if (!initialized) {
        fprintf(stderr, "Call MPI_Init before calling this function.\n");
        MPI_Abort(comm, -1);
    }

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int numberOfNodes = getSlurmNumNodes();
    int warmupRuns = 10;

    FILE* logfile;
    if (logging) {
        if (rank == 0) {
            std::string machineName = MACHINE_NAME;
            std::string filename = "DGEMM_execution_logs-PreDistributed_GEMM-" + machineName + "-COSMA(ScaLAPACK).csv";
            std::string header = "Algo,M,N,K,Mb,Nb,TotalNodes,TotalGPUs,DecompositionTime,ExecutionTime,GatherTime,GFlops,DataLocation";
            logfile = createLogCsv(filename, header);
        }
    }

    double decompBefore = MPI_Wtime();
    pblasDecomposer decomposerA(M, K, Mb, Nb, MPI_COMM_WORLD);
    pblasDecomposer decomposerB(K, N, Mb, Nb, MPI_COMM_WORLD);
    pblasDecomposer decomposerC(M, N, Mb, Nb, MPI_COMM_WORLD);

    scalar_t *localA, *localB, *localC;
    localA = (scalar_t*) malloc(sizeof(scalar_t) * decomposerA.localRows * decomposerA.localColumns);
    localB = (scalar_t*) malloc(sizeof(scalar_t) * decomposerB.localRows * decomposerB.localColumns);
    localC = (scalar_t*) malloc(sizeof(scalar_t) * decomposerC.localRows * decomposerC.localColumns);

    decomposerA.scatterMatrix(0, A, localA);
    decomposerB.scatterMatrix(0, B, localB);
    decomposerC.scatterMatrix(0, C, localC);

    int dRow = decomposerA.dRow;
    int dCol = decomposerA.dCol;
    grid_layout<scalar_t> gridA = block_cyclic_layout<scalar_t>(
                    M, K,
                    Mb, Nb,
                    1, 1, //assume that rank 0 holds matrices
                    M, K,
                    dRow, dCol,
                    'R',
                    0, 0, //assume that rank 0 holds matrices
                    localA,
                    decomposerA.localRows, //llda is rows -> Column Major
                    'C',
                    rank
    );
    
    grid_layout<scalar_t> gridB = block_cyclic_layout<scalar_t>(
                    K, N,
                    Mb, Nb,
                    1, 1, //assume that rank 0 holds matrices
                    K, N,
                    dRow, dCol,
                    'R',
                    0, 0, //assume that rank 0 holds matrices
                    localB,
                    decomposerB.localRows,
                    'C',
                    rank
    );

    grid_layout<scalar_t> gridC = block_cyclic_layout<scalar_t>(
                    M, N,
                    Mb, Nb,
                    1, 1, //assume that rank 0 holds matrices
                    M, N, 
                    dRow, dCol,
                    'R',
                    0, 0, //assume that rank 0 holds matrices
                    localC,
                    decomposerC.localRows,
                    'C',
                    rank
    );
    double decompAfter = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < numberOfRuns; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double executionBefore = MPI_Wtime();
        multiply_using_layout(gridA, gridB, gridC, alpha, beta, TransA, TransB, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double executionAfter = MPI_Wtime();

        double gatherAfter, gatherBefore;
        if (gatherResult) {
            gatherBefore = MPI_Wtime();
            decomposerC.gatherMatrix(0, C, localC);
            gatherAfter = MPI_Wtime();
        }

        /* Write to logfile */
        if (logging) {
            if (rank ==  0) {
                if (i == 0)
                    continue;
                    
                double decompTime = decompAfter - decompBefore;
                double executionTime = executionAfter - executionBefore;
                double gatherTime = gatherAfter - gatherBefore;
                double gflops = calculateGflops(M, N, K, executionTime);
                char csvLine[250];
                sprintf(csvLine, "%s,%lld,%lld,%lld,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%s\n",
                                "Cosma(ScaLAPACK)", M, N, K, Mb, Nb, numberOfNodes, size, decompTime, executionTime, gatherTime, gflops, "host");
                writeLineToFile(logfile, csvLine);
            }
        }
    }   

    if (logging) {
        if (rank == 0) {
            fclose(logfile);
        }
    }

    return;
}

template <typename scalar_t>
void cosmaPreDistributedScalapackGemm(char TransA, char TransB, const long long M, const long long N, const long long K, scalar_t alpha, scalar_t* A, const long long lda,
    scalar_t* B, const long long ldb, scalar_t beta, scalar_t* C, const long long ldc, const int Mb, const int Nb, MPI_Comm comm, const int numberOfRuns, bool logging)
{
    using namespace cosma;
    using namespace costa;

    int initialized;
    MPI_Initialized(&initialized);

    if (!initialized) {
        fprintf(stderr, "Call MPI_Init before calling this function.\n");
        MPI_Abort(comm, -1);
    }

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int numberOfNodes = getSlurmNumNodes();
    int warmupRuns = 10;

    FILE* logfile;
    if (logging) {
        if (rank == 0) {
            std::string machineName = MACHINE_NAME;
            std::string filename = "DGEMM_execution_logs-PreDistributed_GEMM-" + machineName + "-COSMA(ScaLAPACK).csv";
            std::string header = "Algo,M,N,K,Mb,Nb,TotalNodes,TotalGPUs,DecompositionTime,ExecutionTime,GFlops,DataLocation";
            logfile = createLogCsv(filename, header);
        }
    }

    double decompBefore = MPI_Wtime();
    pblasDecomposer decomposerA(M, K, Mb, Nb, MPI_COMM_WORLD);
    pblasDecomposer decomposerB(K, N, Mb, Nb, MPI_COMM_WORLD);
    pblasDecomposer decomposerC(M, N, Mb, Nb, MPI_COMM_WORLD);

    scalar_t *localA, *localB, *localC;
    localA = (scalar_t*) malloc(sizeof(scalar_t) * decomposerA.localRows * decomposerA.localColumns);
    localB = (scalar_t*) malloc(sizeof(scalar_t) * decomposerB.localRows * decomposerB.localColumns);
    localC = (scalar_t*) malloc(sizeof(scalar_t) * decomposerC.localRows * decomposerC.localColumns);

    MatrixInit(localA, decomposerA.localRows, decomposerA.localColumns, 0);
    MatrixInit(localB, decomposerB.localRows, decomposerB.localColumns, 0);
    MatrixInit(localC, decomposerC.localRows, decomposerC.localColumns, 0);

    int dRow = decomposerA.dRow;
    int dCol = decomposerA.dCol;
    grid_layout<scalar_t> gridA = block_cyclic_layout<scalar_t>(
                    M, K,
                    Mb, Nb,
                    1, 1, //assume that rank 0 holds matrices
                    M, K,
                    dRow, dCol,
                    'R',
                    0, 0, //assume that rank 0 holds matrices
                    localA,
                    decomposerA.localRows, //llda is rows -> Column Major
                    'C',
                    rank
    );
    
    grid_layout<scalar_t> gridB = block_cyclic_layout<scalar_t>(
                    K, N,
                    Mb, Nb,
                    1, 1, //assume that rank 0 holds matrices
                    K, N,
                    dRow, dCol,
                    'R',
                    0, 0, //assume that rank 0 holds matrices
                    localB,
                    decomposerB.localRows,
                    'C',
                    rank
    );

    grid_layout<scalar_t> gridC = block_cyclic_layout<scalar_t>(
                    M, N,
                    Mb, Nb,
                    1, 1, //assume that rank 0 holds matrices
                    M, N, 
                    dRow, dCol,
                    'R',
                    0, 0, //assume that rank 0 holds matrices
                    localC,
                    decomposerC.localRows,
                    'C',
                    rank
    );
    double decompAfter = MPI_Wtime();

    for (int i = 0; i < warmupRuns; i++) {
        multiply_using_layout(gridA, gridB, gridC, alpha, beta, TransA, TransB, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < numberOfRuns; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double executionBefore = MPI_Wtime();
        multiply_using_layout(gridA, gridB, gridC, alpha, beta, TransA, TransB, MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        double executionAfter = MPI_Wtime();

        /* Write to logfile */
        if (logging) {
            if (rank ==  0) {
                if (i == 0)
                    continue;
                    
                double decompTime = decompAfter - decompBefore;
                double executionTime = executionAfter - executionBefore;
                double gflops = calculateGflops(M, N, K, executionTime);
                char csvLine[250];
                sprintf(csvLine, "%s,%lld,%lld,%lld,%d,%d,%d,%d,%lf,%lf,%lf,%s\n",
                                "Cosma(ScaLAPACK)", M, N, K, Mb, Nb, numberOfNodes, size, decompTime, executionTime, gflops, "host");
                writeLineToFile(logfile, csvLine);
            }
        }
    }   

    if (logging) {
        if (rank == 0) {
            fclose(logfile);
        }
    }

    return;
}

template <typename scalar_t>
void cosmaPreDistributedOptimalGemm(char TransA, char TransB, const long long M, const long long N, const long long K, scalar_t alpha, scalar_t* A, const long long lda,
    scalar_t* B, const long long ldb, scalar_t beta, scalar_t* C, const long long ldc, MPI_Comm comm, const int numberOfRuns, bool logging)
{
    using namespace cosma;

    int initialized;
    MPI_Initialized(&initialized);

    if (!initialized) {
        fprintf(stderr, "Call MPI_Init before calling this function.\n");
        MPI_Abort(comm, -1);
    }

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int numberOfNodes = getSlurmNumNodes();
    int warmupRuns = 10;

    /* This is set by an env variable called COSMA_CPU_MAX_MEMORY. This value is set in MBs. 
       The actual variable shows the number of matrix elements.
    */
    long long memoryLimit = get_cpu_max_memory<scalar_t>(); 

    bool overlap_comm_and_comp = true;
    Strategy strategy(M, N, K, size, memoryLimit);

    if (overlap_comm_and_comp) 
        strategy.enable_overlapping_comm_and_comp();

    if (rank == 0) {
        std::cout << "Strategy= " << strategy << std::endl;
    }

    FILE* logfile;
    if (logging) {
        if (rank == 0) {
            std::string machineName = MACHINE_NAME;
            std::string filename = "DGEMM_execution_logs-PreDistributed_GEMM-" + machineName + "-COSMA(Optimal).csv";
            std::string header = "Algo,M,N,K,TotalNodes,TotalGPUs,DecompositionTime,ExecutionTime,GFlops,DataLocation";
            logfile = createLogCsv(filename, header);
        }
    }

    /* Warmup GPUs */
    // TODO

    double decompBefore = MPI_Wtime();
    /* Create Cosma Matix classes and Generate Random Data */
    CosmaMatrix<scalar_t> matrixA('A', strategy, rank);
    CosmaMatrix<scalar_t> matrixB('B', strategy, rank);
    CosmaMatrix<scalar_t> matrixC('C', strategy, rank);
    double decompAfter = MPI_Wtime();

    MatrixInit(matrixA.matrix_pointer(), matrixA.matrix_size(), 0);
    MatrixInit(matrixB.matrix_pointer(), matrixB.matrix_size(), 0);
    MatrixInit(matrixC.matrix_pointer(), matrixC.matrix_size(), 0);

    for (int i = 0; i < warmupRuns; i++) {
        multiply(matrixA, matrixB, matrixC, strategy, comm, alpha, beta);
    }
    
    MPI_Barrier(comm);

    /* Call multiply - Result is already gathered on C */
    for (int i = 0; i < numberOfRuns; i++) {
        MPI_Barrier(comm);
        double executionBefore = MPI_Wtime();
        multiply(matrixA, matrixB, matrixC, strategy, comm, alpha, beta);
        MPI_Barrier(comm);
        double executionAfter = MPI_Wtime();
        
        /* Write to logfile */
        if (logging) {
            if (rank ==  0) {
                if (i == 0)
                    continue;
                    
                double decompTime = decompAfter - decompBefore;
                double executionTime = executionAfter - executionBefore;
                double gflops = calculateGflops(M, N, K, executionTime);
                char csvLine[250];
                sprintf(csvLine, "%s,%lld,%lld,%lld,%d,%d,%lf,%lf,%lf,%s\n",
                                "Cosma(Optimal)", M, N, K, numberOfNodes, size, decompTime, executionTime, gflops, "host");
                writeLineToFile(logfile, csvLine);
            }
        }
    }

    if (logging) {
        if (rank == 0) {
            fclose(logfile);
        }
    }

    return;
}

template <typename scalar_t>
void cosmaFullGemmOffloadParalia(char TransA, char TransB, const long long M, const long long N, const long long K, scalar_t alpha, scalar_t* A, const long long lda,
    scalar_t* B, const long long ldb, scalar_t beta, scalar_t* C, const long long ldc, MPI_Comm comm, const int numberOfRuns, bool logging)
{
    /* Use PARALiA Layout and distribute A, B and C from root host to others. Use costa::grid_layout */
}

template <typename scalar_t>
void cosmaPreDistributedParaliaGemm(char TransA, char TransB, const long long M, const long long N, const long long K, scalar_t alpha, scalar_t* A, const long long lda,
    scalar_t* B, const long long ldb, scalar_t beta, scalar_t* C, const long long ldc, MPI_Comm comm, const int numberOfRuns, bool logging)
{
    using namespace cosma;

    return;
}

template <typename scalar_t>
void validateCosmaGEMM(char TransA, char TransB, const long long M, const long long N, const long long K, scalar_t alpha, scalar_t* A, const long long lda,
    scalar_t* B, const long long ldb, scalar_t beta, scalar_t* C, const long long ldc, MPI_Comm comm)
{
    return;
}

template <typename scalar_t>
void cosmaGEMM(char TransA, char TransB, const long long M, const long long N, const long long K, scalar_t alpha, scalar_t* A, const long long lda,
    scalar_t* B, const long long ldb, scalar_t beta, scalar_t* C, const long long ldc, MPI_Comm comm)
{
    using namespace cosma;

    int initialized;
    MPI_Initialized(&initialized);

    if (!initialized) {
        fprintf(stderr, "Call MPI_Init before calling this function.\n");
        MPI_Abort(comm, -1);
    }

    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    long long memoryLimit = get_cpu_max_memory<scalar_t>();

    bool overlap_comm_and_comp = true;
    const Strategy strategy(M, N, K, size, memoryLimit);

    
    if (rank == 0) {
        std::cout << "Strategy= " << strategy << std::endl;
    }

    /* Initialize Matrices */
    /* Distribute Data with current strategy */
    /* Run Multiply */
    MPI_Barrier(comm);
    // multiply(matrixA, matrixB, matrixC, strategy, comm, alpha, beta);
    /* Gather Results to rank=0 */
    
}

template void cosmaPreDistributedOptimalGemm<double>(char TransA, char TransB, const long long M, const long long N, const long long K, double alpha, double* A, const long long lda,
    double* B, const long long ldb, double beta, double* C, const long long ldc, MPI_Comm comm, int numberOfRuns, bool logging);
template void cosmaPreDistributedOptimalGemm<float>(char TransA, char TransB, const long long M, const long long N, const long long K, float alpha, float* A, const long long lda,
    float* B, const long long ldb, float beta, float* C, const long long ldc, MPI_Comm comm, int numberOfRuns, bool logging);

template void cosmaPreDistributedScalapackGemm<double>(char TransA, char TransB, const long long M, const long long N, const long long K, double alpha, double* A, const long long lda,
    double* B, const long long ldb, double beta, double* C, const long long ldc, const int Mb, const int Nb, MPI_Comm comm, int numberOfRuns, bool logging);

template void cosmaFullGemmOffloadScalapack<double>(char TransA, char TransB, const long long M, const long long N, const long long K, double alpha, double* A, const long long lda,
    double* B, const long long ldb, double beta, double* C, const long long ldc, int Mb, int Nb, MPI_Comm comm, const int numberOfRuns, bool logging, bool gatherResult);
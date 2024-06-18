#include <cosmaWrapped.hpp>

template <typename scalar_t>
void cosmaFullGemmOffloadScalapack(char TransA, char TransB, const long long M, const long long N, const long long K, scalar_t alpha, scalar_t* A, const long long lda,
    scalar_t* B, const long long ldb, scalar_t beta, scalar_t* C, const long long ldc, MPI_Comm comm, const int numberOfRuns, bool logging)
{
    /* Use ScaLAPACK Layout and distribute A, B and C from root host to others. Use pdgemm wrapper of COSMA */
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
    /* Generate Random data, but use PARALiA Layout. Check COSTA source code on how to translate PARALiA layout to costa::grid_layout */
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

    /* This is set by an env variable called COSMA_CPU_MAX_MEMORY. This value is set in MBs. 
       The actual variable shows the number of matrix elements (στοιχεία πίνακα, δεν μου έρχεται τώρα).
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
}

template <typename scalar_t>
void cosmaPreDistributedScalapackGemm(char TransA, char TransB, const long long M, const long long N, const long long K, scalar_t alpha, scalar_t* A, const long long lda,
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
    int totalGPUs = size;

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
            std::string filename = "DGEMM_execution_logs-PreDistributedScalapack_GEMM-" + machineName + "-COSMA.csv";
            std::string header = "Algo,M,N,K,TileRows,TileColumns,dRow,dCol,TotalNodes,TotalGPUs,DecompositionTime,ExecutionTime,GatherTime,GFlops,DataLocation";
            logfile = createLogCsv(filename, header);
        }
    }

    /* Copy matrix (ends in barrier)*/
    CosmaMatrix<scalar_t> matrixA('A', strategy, rank);
    CosmaMatrix<scalar_t> matrixB('B', strategy, rank);
    CosmaMatrix<scalar_t> matrixC('C', strategy, rank);

    /* TODO: Create a method to generate a matrix for size=M*N (one-dimensional, without stride) */
    fill_int(matrixA.matrix_pointer(), matrixA.matrix_size());
    fill_int(matrixB.matrix_pointer(), matrixB.matrix_size());
    fill_int(matrixC.matrix_pointer(), matrixC.matrix_size());

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
                double executionTime = executionAfter - executionBefore;
                double gflops = calculateGflops(M, N, K, executionTime);
                char csvLine[200];
                sprintf(csvLine, "%s, ");
            }
        }
    }

    if (logging) {
        if (rank == 0) {
            fclose(logfile);
        }
    }
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
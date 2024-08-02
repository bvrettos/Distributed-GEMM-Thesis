#include <distributedParalia.hpp>

void preDistributedSequentialParaliaGemm(char TransA,  char TransB, const long long M, const long long N, const long long K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC, int numberOfRuns, bool logging, bool gatherResults, int aLoc, int bLoc, int cLoc)
{
    /* Check if MPI has been initialized */
    int initializedMPI;
    MPI_Initialized(&initializedMPI);

    if (!initializedMPI) {
        std::cerr << "ERROR: MPI has not been initialized. Call MPI_Init before calling this function" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -2);
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Create logfile */
    FILE* logfile;
    if (logging) {
        if (rank == 0) {
            std::string machineName = MACHINE_NAME;
            std::string filename = "DGEMM_execution_logs-PreDistributed_GEMM-" + machineName + "-PARALIA_Sequential.csv";
            std::string header = "Algo,M,N,K,dRow,dCol,TotalNodes,TotalGPUs,DecompositionTime,ExecutionTime,GatherTime,GFlops,ALoc,BLoc,CLoc";
            logfile = createLogCsv(filename, header);
        }
    }
    
    BlockSequentialDecomposer Decomposer(M, N, K, MPI_COMM_WORLD);
    int totalGPUs = Decomposer.size * 4;
    int dRow = Decomposer.dRow;
    int dCol = Decomposer.dCol;
    int numberOfNodes = getSlurmNumNodes();
    
    long long int localM = Decomposer.localM;
    long long int localN = Decomposer.localN;
    long long int localK = Decomposer.localK;

    long long int llda = localM;
    long long int lldb = localK;
    long long int lldc = localM;

    double *localA, *localB, *localC;
    localA = (double*) CHLMalloc(localM * localK * sizeof(double), aLoc, 0);
	localB = (double*) CHLMalloc(localK * localN * sizeof(double), bLoc, 0);
	localC = (double*) CHLMalloc(localM * localN * sizeof(double), cLoc, 1);

    /* Generate Data */
    CHLVecInit(localA, localM * localK, 42, aLoc);
    CHLVecInit(localB, localK * localN, 17, bLoc);
    CHLVecInit(localC, localM * localN, 1337, cLoc);
    CHLSyncCheckErr();

    /* Warmup Runs */
    for (int i = 0; i < 10; i ++) {
        PARALiADgemm(TransA, TransB, localM, localN, localK, alpha, localA, llda, localB, lldb, beta, localC, lldc);
    }

    /* Actual Runs, use Barrier before running clocks to be more precise. */
    for (int i = 0; i < numberOfRuns; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double executionStart = MPI_Wtime();
        PARALiADgemm(TransA, TransB, localM, localN, localK, alpha, localA, llda, localB, lldb, beta, localC, lldc);
        MPI_Barrier(MPI_COMM_WORLD);
        double executionEnd = MPI_Wtime();

        double gatherBefore, gatherAfter;
        if (gatherResults) {
            gatherBefore = MPI_Wtime();
            // Decomposer.gatherResult(0, C, localC);
            gatherAfter = MPI_Wtime();
        }

        if (logging) {
            if (rank == 0) {
                double executionTime = executionEnd-executionStart;
                double gflops = (2 * M * N * K * 1e-9) / executionTime;
                double gatherTime = gatherAfter-gatherBefore;
                char csvLine[300];
                sprintf(csvLine, "%s,%lld,%lld,%lld,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%d,%d,%d\n", 
                    "PARALiA-Sequential", M, N, K, dRow, dCol, numberOfNodes, totalGPUs, 0.0, executionTime, gatherTime, gflops, aLoc, bLoc, cLoc);
                writeLineToFile(logfile, csvLine);
            }
        }
    }

    /* Free everything not needed */
    CHLSyncCheckErr();
    CHLFree(localA, localM * localK * sizeof(double), aLoc);
    CHLFree(localB, localN * localK * sizeof(double), bLoc);
    CHLFree(localC, localM * localN * sizeof(double), cLoc);

    if (logging) {
        if (rank == 0)
            fclose(logfile);
    }

    return;
}

void paraliaFullGemmOffloadSequential(char TransA,  char TransB, const long long M, const long long N, const long long K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC, int numberOfRuns, bool logging, bool warmup, int aLoc, int bLoc, int cLoc)
{

    /* Check if MPI has been initialized */
    int initializedMPI;
    MPI_Initialized(&initializedMPI);

    if (!initializedMPI) {
        std::cerr << "ERROR: MPI has not been initialized. Call MPI_Init before calling this function" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -2);
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Create logfile */
    FILE* logfile;
    if (logging) {
        if (rank == 0) {
            std::string machineName = MACHINE_NAME;
            std::string filename = "DGEMM_execution_logs-FullOffload_GEMM-" + machineName + "-PARALIA_Sequential.csv";
            std::string header = "Algo,M,N,K,dRow,dCol,TotalNodes,TotalGPUs,DecompositionTime,ExecutionTime,GatherTime,GFlops,ALoc,BLoc,CLoc";
            logfile = createLogCsv(filename, header);
        }
    }
    
    BlockSequentialDecomposer Decomposer(M, N, K, MPI_COMM_WORLD);
    int totalGPUs = Decomposer.size*4;
    int dRow = Decomposer.dRow;
    int dCol = Decomposer.dCol;
    int numberOfNodes = getSlurmNumNodes();

    /* Local matrices are also Column-Major */
    long long int localM = Decomposer.localM;  
    long long int localN = Decomposer.localN;
    long long int localK = Decomposer.localK;

    long long int llda = localM;
    long long int lldb = localK;
    long long int lldc = localM;

    double *localA, *localB, *localC;
    localA = (double*) CHLMalloc(localM * localK * sizeof(double), aLoc, 0);
	localB = (double*) CHLMalloc(localK * localN * sizeof(double), bLoc, 0);
	localC = (double*) CHLMalloc(localM * localN * sizeof(double), cLoc, 1);

    double decompositionStart = MPI_Wtime();
    Decomposer.deliverMatrix(A, B, C, &localA, &localB, &localC);
    double decompositionEnd = MPI_Wtime();

    CHLSyncCheckErr();

    /* Warmup Runs */
    if (warmup) {
        for (int i = 0; i < 10; i ++) {
            PARALiADgemm(TransA, TransB, localM, localN, localK, alpha, localA, llda, localB, lldb, beta, localC, lldc);
        }
    }

    /* Actual Runs, use Barrier before running clocks to be more precise. */
    for (int i = 0; i < numberOfRuns; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double executionStart = MPI_Wtime();
        PARALiADgemm(TransA, TransB, localM, localN, localK, alpha, localA, llda, localB, lldb, beta, localC, lldc);
        MPI_Barrier(MPI_COMM_WORLD);
        double executionEnd = MPI_Wtime();

        double gatherBefore, gatherAfter;
        gatherBefore = MPI_Wtime();
        Decomposer.gatherResult(0, C, localC);
        gatherAfter = MPI_Wtime();

        if (logging) {
            if (rank == 0) {
                double executionTime = executionEnd-executionStart;
                double gflops = (2 * M * N * K * 1e-9) / executionTime;
                double gatherTime = gatherAfter-gatherBefore;
                double decompositionTime = decompositionEnd - decompositionStart;
                char csvLine[300];
                sprintf(csvLine, "%s,%lld,%lld,%lld,%d,%d,%d,%d,%lf,%lf,%lf,%lf,%d,%d,%d\n", 
                    "PARALiA-Sequential", M, N, K, dRow, dCol, numberOfNodes, totalGPUs, decompositionTime, executionTime, gatherTime, gflops, aLoc, bLoc, cLoc);
                writeLineToFile(logfile, csvLine);
            }
        }
    }

    /* Free everything not needed */
    CHLSyncCheckErr();
    CHLFree(localA, localM * localK * sizeof(double), aLoc);
    CHLFree(localB, localN * localK * sizeof(double), bLoc);
    CHLFree(localC, localM * localN * sizeof(double), cLoc);

    if (logging) {
        if (rank == 0)
            fclose(logfile);
    }

    return;
}


void paraliaPreDistributedScalapackGemm(char TransA,  char TransB, const long long M, const long long N, const long long K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC, int numberOfRuns, bool logging, bool gatherResults, int aLoc, int bLoc, int cLoc)
{

}

void paraliaFullGemmOffloadCyclic(char TransA,  char TransB, const long long M, const long long N, const long long K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC, int numberOfRuns, bool logging, bool gatherResults, int aLoc, int bLoc, int cLoc)
{

}
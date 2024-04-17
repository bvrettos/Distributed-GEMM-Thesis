#include "mpiBLAS_wrappers.hpp"

double PARALiA_MPI_Dgemm(char TransA,  char TransB, const long long M, const long long N, const long long K,
  double alpha, double* A, long int ldA, double* B, long int ldB, double beta, double* C,
  long int ldC)
{
    /* Check if MPI has been initialized */
    int initializedMPI;
    MPI_Initialized(&initializedMPI);

    if (!initializedMPI) {
        std::cerr << "ERROR: MPI has not been initialized. Call MPI_Init before calling this function" << std::endl;
        exit(-2);
    }

    /* Create a communicator only for processes that can have access to a GPU */
    // MPI_Comm gpuCommunicator = createGPUCommunicator();
    MPI_Comm problemCommunicator = MPI_COMM_WORLD;

    int rank;
    MPI_Comm_rank(problemCommunicator, &rank);

    /* Create logfile */
    FILE* logfile;
    if (rank == 0) {
        std::string machineName = MACHINE_NAME;
        std::string filename = "DGEMM_execution_logs-" + machineName + "-PARALIA_Sequential.csv";
        std::string header = "Algo,M,N,K,TotalNodes,TotalGPUs,DecompositionTime,ExecutionTime,GFlops";
        logfile = createLogCsv(filename, header);
    }
    
    GEMM_BlockSequentialDecomposer Decomposer(M, N, K, problemCommunicator);
    
    double *localA, *localB, *localC;

    long long int localM = Decomposer.localM;
    long long int localN = Decomposer.localN;
    long long int localK = Decomposer.localK;

    long long int llda = localM;
    long long int lldb = localK;
    long long int lldc = localM;

    localA = (double*) malloc(localK * localM * sizeof(double));
	localB = (double*) malloc(localN * localK * sizeof(double));
	localC = (double*) malloc(localM * localN * sizeof(double));

    double decompositionStart = MPI_Wtime();
    Decomposer.scatterMatrices(A, B, C, localA, localB, localC);
    double decompositionTime = MPI_Wtime() - decompositionStart;

    for (int i = 0; i < 1; i++) {
        double executionStart = MPI_Wtime();
        PARALiADgemm('N', 'N', localM, localN, localK, alpha, localA, llda, localB, lldb, beta, localC, lldc);
        double executionEnd = MPI_Wtime();

        if (rank == 0) {
            double executionTime = executionEnd-executionStart;
            double gflops = (2 * M * N * K * 1e-9) / executionTime;
            int totalGPUs = Decomposer.communicatorSize;
            int numberOfNodes = 1;

            char csvLine[150];
            sprintf(csvLine, "%s,%lld,%lld,%lld,%d,%d,%lf,%lf,%lf\n", "PARALiA-Sequential", M, N, K, numberOfNodes, totalGPUs, decompositionTime, executionTime, gflops);
            writeLineToFile(logfile, csvLine);
        }
    }

    /* Return C to main process */
    Decomposer.gatherResult(C, localC);

    /* Free everything not needed */
    free(localA);
    free(localB);
    free(localC);

    return 0;
}
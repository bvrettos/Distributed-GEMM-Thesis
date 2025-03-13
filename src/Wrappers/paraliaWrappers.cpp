#include <paraliaWrappers.hpp>

template <typename scalar_t>
void singleNodeParaliaGemm(char TransA, char TransB, const long long M, const long long N, const long long K, scalar_t alpha, scalar_t* A, long long int ldA,
    scalar_t* B, long long int ldB, scalar_t beta, scalar_t* C, long long ldC, int numberOfRuns, bool logging, int aLoc, int bLoc, int cLoc)
{   
    FILE* logfile;
    /* No need for MPI initializations or rank, size getters. Just a simple PARALiA wrapper */
    if (logging) {
        std::string machineName = MACHINE_NAME;
        std::string filename = "DGEMM_execution_logs-PreDistributed_GEMM-" + machineName + "-PARALiA.csv";
        std::string header = "Algo,M,N,K,TotalGPUs,ExecutionTime,GFlops,ALoc,BLoc,CLoc";
        logfile = createLogCsv(filename, header);
    }

    A = (scalar_t*) CHLMalloc(sizeof(scalar_t) * M * K, aLoc, 0);
    B = (scalar_t*) CHLMalloc(sizeof(scalar_t) * K * N, bLoc, 0);
    C = (scalar_t*) CHLMalloc(sizeof(scalar_t) * M * N, cLoc, 1);

    CHLVecInit(A, M * K, 42, aLoc);
    CHLVecInit(B, K * N, 17, bLoc);
    CHLVecInit(C, M * N, 1337, cLoc);

    CHLSyncCheckErr();

    for (int i = 0; i < numberOfRuns; i++) {
        double executionStart = csecond();
        PARALiADgemm(TransA, TransB, M, N, K, alpha, A, ldA, B, ldB, beta, C, ldC); 
        double executionEnd = csecond(); // Dgemm call is blocking, no need to sync

        if (logging) {
            if (i == 0)
                continue;
            double executionTime = executionEnd - executionStart;
            double gflops = calculateGflops(M, N, K, executionTime);
            char csvLine[150];
            sprintf(csvLine, "%s,%lld,%lld,%lld,%d,%lf,%lf,%d,%d,%d\n", 
                "PARALiA", M, N, K, 4, executionTime, gflops, aLoc, bLoc, cLoc);
            writeLineToFile(logfile, csvLine);
        }
    }
    
    for (int i = 0; i < CHL_MEMLOCS; i++) PARALiADevCacheFree((i));

    CHLSyncCheckErr();
    CHLFree(A, M * K * sizeof(scalar_t), aLoc);
    CHLFree(B, K * N * sizeof(scalar_t), bLoc);
    CHLFree(C, M * N * sizeof(scalar_t), cLoc);

    if (logging)
        fclose(logfile);

    return;
}

template void singleNodeParaliaGemm<double>(char TransA, char TransB, const long long M, const long long N, const long long K, double alpha, double* A, long long int ldA,
    double* B, long long int ldB, double beta, double* C, long long ldC, int numberOfRuns, bool logging, int aLoc, int bLoc, int cLoc);
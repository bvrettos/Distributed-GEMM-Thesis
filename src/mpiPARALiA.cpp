#include "mpiBLAS_wrappers.hpp"
#include "cudaCommunicator.hpp"


double PARALiA_MPI_Dgemm(char TransA,  char TransB, long int M, long int N, long int K,
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
    MPI_Comm gpuCommunicator = createGPUCommunicator();
    MPI_Comm problemCommunicator = gpuCommunicator;

    GEMM_BlockSequentialDecomposer Decomposer(M, N, K, problemCommunicator);

    int rank = Decomposer.rank;

    double *localA, *localB, *localC;

    int localM = Decomposer.localM;
    int localN = Decomposer.localN;
    int localK = Decomposer.localK;

    localA = (double*) malloc(localK * localM*sizeof(double));
	localB = (double*) malloc(localN * localK*sizeof(double));
	localC = (double*) malloc(localM * localN*sizeof(double));

    Decomposer.scatterMatrices(A, B, C, localA, localB, localC);

    /* PARALIA: Create ATC */
    ATC_p predefControlValues = new ATC();
    predefControlValues->cache_limit = -1;
    predefControlValues->T = -1;
    predefControlValues->active_unit_num = 1;
    predefControlValues->active_unit_id_list[0] = 1;

    PARALiADgemmControled('N', 'N', localM, localN, localK, alpha, localA,
     localK, localB, localN, beta, localC, localN, predefControlValues);

    /* Return C to main process */
    Decomposer.gatherResult(C, localC);

    /* Free everything not needed */
    free(localA);
    free(localB);
    free(localC);

    return 0;
}
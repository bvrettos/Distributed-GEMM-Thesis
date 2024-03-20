#include "2DBlockSequentialDecomposition.hpp"

GEMM_BlockSequentialDecomposer::GEMM_BlockSequentialDecomposer(int M, int N, int K, MPI_Comm communicator) : M(M), N(N), K(K), GEMM_Communicator(communicator)
{
    MPI_CHECK(MPI_Comm_rank(GEMM_Communicator, &rank));
    MPI_CHECK(MPI_Comm_size(GEMM_Communicator, &communicatorSize));

    numberOfDevices = communicatorSize;

    calculateVirtualDeviceGrid();

    localM = M/dRow;
    localN = N/dCol;
    localK = K;

    allocateMPIDatatypes();
    calculateScatterValues();
}

void GEMM_BlockSequentialDecomposer::calculateVirtualDeviceGrid()
{
    int Px = std::sqrt(numberOfDevices);
    int Py = Px;

    /* If less than 4 devices */
    if (Px == 0) {
        Py = numberOfDevices;
        Px = 1;
    }
    /* If more than 4 devices, find the most square decomposition */
    int counter;
    for (counter = Px; counter > 0; --counter) 
        if (numberOfDevices % counter == 0) break;
    
    if (counter==0) {
        Px = numberOfDevices;
        Py = 1;
    }
    else {
        Px = counter;
        Py = numberOfDevices/counter;
    }

    dRow = Py;
    dCol = Px;

#ifdef DEBUG
    printf("dRow: %d - dCol: %d\n", dRow, dCol);
#endif
}

void GEMM_BlockSequentialDecomposer::allocateMPIDatatypes()
{
    /* Local 2D block MPI Definitions */
    MPI_Type_vector(localM, localK, localK, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &localBlockA);
    MPI_Type_commit(&localBlockA);

    MPI_Type_vector(localK, localN, localN, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &localBlockB);
    MPI_Type_commit(&localBlockB);

    MPI_Type_vector(localM, localN, localN, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &localBlockC);
    MPI_Type_commit(&localBlockC);

    /* Global 2D block MPI Definitions */
    MPI_Type_vector(localM, localK, K, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &globalBlockA);
    MPI_Type_commit(&globalBlockA);

    MPI_Type_vector(localK, localN, N, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &globalBlockB);
    MPI_Type_commit(&globalBlockB);

    MPI_Type_vector(localM, localN, N, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &globalBlockC);
    MPI_Type_commit(&globalBlockC);
}

void GEMM_BlockSequentialDecomposer::calculateScatterValues()
{
    if (rank == 0) {
        scatterOffsetA = (int*) malloc(numberOfDevices * sizeof(int));
        scatterCountA = (int*) malloc(numberOfDevices * sizeof(int));

        scatterOffsetB = (int*) malloc(numberOfDevices * sizeof(int));
        scatterCountB = (int*) malloc(numberOfDevices * sizeof(int));

        scatterOffsetC = (int*) malloc(numberOfDevices * sizeof(int));
        scatterCountC = (int*) malloc(numberOfDevices * sizeof(int));

        /* C Scattering 2D Sequential Decomp*/
        for (int i = 0; i < dRow; i++) {   
            for (int j = 0; j < dCol; j++) {
                scatterCountC[i*dCol + j] = 1;
                scatterOffsetC[i*dCol + j] = localM * localN * dCol * i + localN * j;
            }
        }

        /* A Scattering 1D Decomp*/
        for (int i = 0; i < dRow; i++) {
            for (int j = 0; j < dCol; j++) {
                scatterCountA[i*dCol + j] = 1;
                scatterOffsetA[i*dCol + j] = localM*localK*i;
            }
        }

        /* B Scattering 1D Decomp*/
        for (int i = 0; i < dCol; i++) {
            for (int j = 0; j < dRow; j++) {
                scatterCountB[j*dCol + i] = 1;
                scatterOffsetB[j*dCol + i] = localN*i;
            }
        }

        #ifdef DEBUG
            for (int k = 0; k < size; k++) {
                std::cout << "ScatteroffsetA: " << scatterOffsetA[k] << std::endl;
                std::cout << "ScatteroffsetB: " << scatterOffsetB[k] << std::endl;
                std::cout << "ScatteroffsetC: " << scatterOffsetC[k] << std::endl;
                std::cout << std::endl;
            }
        #endif
    }
}

void GEMM_BlockSequentialDecomposer::scatterMatrices(double* A, double* B, double* C, double* localA, double* localB, double* localC)
{
    /* Scatter Matrices */
    MPI_CHECK(MPI_Scatterv(A, scatterCountA, scatterOffsetA, globalBlockA, localA, 1, localBlockA, 0, GEMM_Communicator));
    MPI_CHECK(MPI_Scatterv(B, scatterCountB, scatterOffsetB, globalBlockB, localB, 1, localBlockB, 0, GEMM_Communicator));
    MPI_CHECK(MPI_Scatterv(C, scatterCountC, scatterOffsetC, globalBlockC, localC, 1, localBlockC, 0, GEMM_Communicator));
}

void GEMM_BlockSequentialDecomposer::gatherResult(double* C, double* localC)
{
    MPI_CHECK(MPI_Gatherv(localC, 1, localBlockC, C, scatterCountC, scatterOffsetC, globalBlockC, 0, GEMM_Communicator));
}

GEMM_BlockSequentialDecomposer::~GEMM_BlockSequentialDecomposer()
{
    if (rank == 0) {
        free(scatterCountA);
        free(scatterCountB);
        free(scatterCountC);

        free(scatterOffsetA);
        free(scatterOffsetB);
        free(scatterOffsetC);
    }

    MPI_Type_free(&localBlockA);
    MPI_Type_free(&localBlockB);
    MPI_Type_free(&localBlockC);

    MPI_Type_free(&globalBlockA);
    MPI_Type_free(&globalBlockB);
    MPI_Type_free(&globalBlockC);
}
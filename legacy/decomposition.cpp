#include "decomposition.hpp"

blockSequentialDecomposer::blockSequentialDecomposer(const int M, const int N, const int K, MPI_Comm communicator) : M(M), N(N), K(K), communicator(communicator) {
    numDevices = 0;
}   

void blockSequentialDecomposer::calculateGridDimensions()
{
    MPI_Comm_size(communicator, &numDevices);

    int Px = std::sqrt(numDevices);
    int Py = Px;

    /* If less than 4 devices */
    if (Px == 0) {
        Py = numDevices;
        Px = 1;
    }

    /* If more than 4 devices, find the most square decomposition */
    int counter;
    for (counter = Px; counter > 0; --counter) 
        if (numDevices % counter == 0) break;
    
    if (counter==0) {
        Px = numDevices;
        Py = 1;
    }
    else {
        Px = counter;
        Py = numDevices/counter;
    }

    
    dRow = Py;
    dCol = Px;

    localK = K;
    localM = M/dRow;
    localN = N/dCol;
}

void blockSequentialDecomposer::allocateMPIDatatypes()
{
    /* Local 2D block MPI Definitions */
    MPI_Type_vector(localM, localK, localK, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &blockA);
    MPI_Type_commit(&blockA);

    MPI_Type_vector(localK, localN, localN, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &blockB);
    MPI_Type_commit(&blockB);

    MPI_Type_vector(localM, localN, localN, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &blockC);
    MPI_Type_commit(&blockC);

    /* Global 2D block MPI Definitions */
    MPI_Type_vector(localM, localK, K, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &globalA);
    MPI_Type_commit(&globalA);

    MPI_Type_vector(localK, localN, N, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &globalB);
    MPI_Type_commit(&globalB);

    MPI_Type_vector(localM, localN, N, MPI_DOUBLE, &dummy);
    MPI_Type_create_resized(dummy, 0, sizeof(double), &globalC);
    MPI_Type_commit(&globalC);
}

void blockSequentialDecomposer::calculateScatterValues()
{
    int rank = -1;
    MPI_CHECK(MPI_Comm_rank(communicator, &rank));

    if (rank == 0) {
        scatterOffsetA = (int*) malloc(numDevices*sizeof(int));
        scatterCountA = (int*) malloc(numDevices*sizeof(int));

        scatterOffsetB = (int*) malloc(numDevices*sizeof(int));
        scatterCountB = (int*) malloc(numDevices*sizeof(int));

        scatterOffsetC = (int*) malloc(numDevices*sizeof(int));
        scatterCountC = (int*) malloc(numDevices*sizeof(int));

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

void blockSequentialDecomposer::scatterMatrices(double* A, double* B, double* C, double* localA, double* localB, double* localC)
{
    /* Allocate local Matrices */
    localA = (double *) malloc(sizeof(double)*localM*localK);
    localB = (double *) malloc(sizeof(double)*localK*localN);
    localC = (double *) malloc(sizeof(double)*localM*localN);

    /* Scatter Matrices */
    MPI_CHECK(MPI_Scatterv(A, scatterCountA, scatterOffsetA, globalA, localA, 1, blockA, 0, communicator));
    MPI_CHECK(MPI_Scatterv(B, scatterCountB, scatterOffsetB, globalB, localB, 1, blockB, 0, communicator));
    MPI_CHECK(MPI_Scatterv(C, scatterCountC, scatterOffsetC, globalC, localC, 1, blockC, 0, communicator));
}

void blockSequentialDecomposer::getDecompositionValues(int* localM, int* localN, int* localK)
{
    *localM = this->localM;
    *localN = this->localN;
    *localK = this->localK;
}

void blockSequentialDecomposer::gatherResult(double* C, double* localC)
{
    MPI_Gatherv(localC, 1, blockC, C, scatterCountC, scatterOffsetC, globalC, 0, communicator);
}

blockSequentialDecomposer::~blockSequentialDecomposer()
{

}
#include "pblasGEMM.hpp"

// #define VALIDATE

void pblasDgemm(char* TransA, char* TransB, int M, int N, int K, double alpha, double* A, int lda, double* B, int ldb, double beta,double* C, int ldc, int Mb, int Nb, int dRow, int dCol)
{       
    /* Decompose Matrices */
    pblasDecomposer decomposerA(M, K, Mb, Nb, dRow, dCol, A, MPI_COMM_WORLD);
    pblasDecomposer decomposerB(K, N, Mb, Nb, dRow, dCol, B, MPI_COMM_WORLD);
    pblasDecomposer decomposerC(M, N, Mb, Nb, dRow, dCol, C, MPI_COMM_WORLD);

    const int64_t llda = decomposerA.localRows;
    const int64_t loc_n_a = decomposerA.localColumns;

    const int64_t lldb = decomposerB.localRows;
    const int64_t loc_n_b = decomposerB.localColumns;

    const int64_t lldc = decomposerC.localRows;
    const int64_t loc_n_c = decomposerC.localColumns;

    double *localA, *localB, *localC;
    localA = decomposerA.localMatrix;
    localB = decomposerB.localMatrix;
    localC = decomposerC.localMatrix;

    int cblacsContext = decomposerC.cblacsContext;

    int desca[9];
    int descb[9];
    int descc[9];
    int rsrc = 0;
    int csrc = 0;
    int info;

    descinit_(desca, &M, &K, &Mb, &Nb, &rsrc, &csrc, &cblacsContext, &Mb, &info);
    descinit_(descb, &K, &N, &Mb, &Nb, &rsrc, &csrc, &cblacsContext, &Mb, &info);
    descinit_(descc, &M, &N, &Mb, &Nb, &rsrc, &csrc, &cblacsContext, &Mb, &info);

    int ia = 1, ja = 1, ib = 1, jb = 1, ic = 1, jc = 1;

    double t1 = MPI_Wtime();

    pdgemm_("N", "N", &M, &N, &K, &alpha, localA, &ia, &ja, desca, localB, &ib, &jb, descb,
          &beta, localC, &ic, &jc, descc);

    double t2 = MPI_Wtime();

    int rank = decomposerC.rank;
    if (rank == 0)
        printf("Comp time: %lf\n", t2-t1);

    decomposerC.gatherMatrix();

    return;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int M, N, K;
    int lda, ldb, ldc;
    M = 512;
    N = 512;
    K = 512;

    double alpha, beta;
    alpha = 0.213;
    beta = 1.329;

    double *A, *B, *C, *referenceC;

    if (rank == 0) {
        A = (double*) malloc(sizeof(double) * M * K);
        B = (double*) malloc(sizeof(double) * N * K);
        C = (double*) malloc(sizeof(double) * M * N);

        generateMatrixColumnMajor(A, M, K);
        generateMatrixColumnMajor(B, N, K);
        generateMatrixColumnMajor(C, M, N);

        referenceC = copyMatrix(C, M, N);
    }

    lda = M;
    ldb = K;
    ldc = M;

    double t1 = MPI_Wtime();

    pblasDgemm("N", "N", M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, 32, 32, 2 ,2);
    double t2 = MPI_Wtime();

    if (rank == 0)
        printf("Elapsed Time: %lf \n",t2-t1);

    if (rank == 0) {
        #ifdef VALIDATE
            cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, referenceC, ldc);
            Dtest_equality(C, referenceC, M*N);
        #endif
    }

    MPI_Finalize();

    return 0;
}
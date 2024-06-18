#include <paraliaWrappers.hpp>

int main(int argc, char* argv[])
{
    long long M, N, K;
    double alpha, beta;
    int numberOfRuns;
    char TransposeA, TransposeB;
    int aLoc, bLoc, cLoc;

    if (argc < 11) {
        std::cerr << "Usage ./singleNodeParaliaBenchmark {TransA}{TransB} M N K alpha beta A_loc B_loc C_loc numberOfRuns" << std::endl;
        exit(-1);
    }

    TransposeA = argv[1][0];
    TransposeB = argv[1][1];
    M = atoi(argv[2]);
    N = atoi(argv[3]);
    K = atoi(argv[4]);
    alpha = atof(argv[5]);
    beta = atof(argv[6]);
    aLoc = atoi(argv[7]);
    bLoc = atoi(argv[8]);
    cLoc = atoi(argv[9]);
    numberOfRuns = atoi(argv[10]);

    long long lda, ldb, ldc;
    lda = M;
    ldb = K;
    ldc = M;

    bool logging = true;
    double *A, *B, *C;
    singleNodeParaliaGemm(TransposeA, TransposeB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc, numberOfRuns, logging, aLoc, bLoc, cLoc);

    return 0;
}
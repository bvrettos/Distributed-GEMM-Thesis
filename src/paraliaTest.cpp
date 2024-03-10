#include "matrix.hpp"
#include <PARALiA.hpp>
#include "cmatrix.h"
#include "cblas.h"
#include <cstdio>
#include <typeinfo>
#include <float.h>
#include <curand.h>

int main(int argc, char** argv)
{
    int M, N, K;

    M = 4096;
    N = 4096;
    K = 4096;

    double alpha, beta;
    alpha = 1.0;
    beta = 1.0;

    short A_loc, B_loc, C_loc;
    A_loc = -1;
    B_loc = -1;
    C_loc = -1;

    double *A, *B, *C;
	// allocate in device if loc = 0, otherwise allocate in pinned memory for benchmarks
	A = (double*) CoCoMalloc(M * K*sizeof(double), A_loc, 0);
	B = (double*) CoCoMalloc(N * K*sizeof(double), B_loc, 0);
	C = (double*) CoCoMalloc(M * N*sizeof(double), C_loc, 1);

	CoCoVecInit(A, K * M, 42, A_loc);
	CoCoVecInit(B, K * N, 43, B_loc);
	CoCoVecInit(C, M * N, 44, C_loc);

    long int ldA, ldB, ldC = M;

    /* PARALiA test */
    ATC_p predefControlValues = new ATC();
    predefControlValues->cache_limit = -1;
    predefControlValues->T = -1;
    predefControlValues->active_unit_num = 2;
    predefControlValues->active_unit_id_list[0] = 0;
    predefControlValues->active_unit_id_list[1] = 1;

    PARALiADgemmControled('N', 'N', M, N, K, alpha, A, K, B, N, beta, C, N, predefControlValues);
    //cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A, K, B, N , beta, C, N);

    return 0;
}
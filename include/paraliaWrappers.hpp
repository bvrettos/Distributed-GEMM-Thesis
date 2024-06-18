#ifndef PARALIA_WRAPPERS_HPP
#define PARALIA_WRAPPERS_HPP

#include "PARALiA.hpp"
#include "backend_wrappers.hpp"
#include "logging.hpp"
#include "cmatrix.h"
#include "errorHandling.hpp"
#include "generalUtilities.hpp"

template <typename scalar_t>
void singleNodeParaliaGemm(char TransA, char TransB, const long long M, const long long N, const long long K, scalar_t alpha, scalar_t* A, long long int ldA,
    scalar_t* B, long long int ldB, scalar_t beta, scalar_t* C, long long ldC, int numberOfRuns, bool logging, int aLoc, int bLoc, int cLoc);

#endif
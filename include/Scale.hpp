#ifndef SCALE_HPP
#define SCALE_HPP

#include <DistributedMatrix.hpp>
#include <Tile.hpp>
#include <cublas_v2.h>
#include <cblas.h>
#include <typeinfo>

template <typename scalar_t>
void scaleMatrix(scalar_t scale, DistributedMatrix<scalar_t>& matrix);

template <typename scalar_t>
void scaleTile(scalar_t scale, Tile<scalar_t>& tile);

/* Internal Calls */
cublasStatus_t cublasScale(cublasHandle_t handle, int n, float alpha, float* x, int incx);
cublasStatus_t cublasScale(cublasHandle_t handle, int n, double alpha, double* x, int incx);

void cblasScale(int n, float alpha, float* x, int incx);
void cblasScale(int n, double alpha, double* x, int incx);

#endif 
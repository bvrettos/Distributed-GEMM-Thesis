#include <Scale.hpp>

/* Need to create wrappers for both cblas and cublas scale functions */
template <typename scalar_t>
void scaleTile(scalar_t scale, Tile<scalar_t>& tile)
{
    MemoryLocation location = tile.getLocation();
    int64_t m = tile.getRows();
    int64_t n = tile.getColumns();
    scalar_t* matrix = tile.getDataPointer();

    if (location == MemoryLocation::Host) {
        /* CPU Scale*/
        cblasScale(m*n, scale, matrix, 1);
    }

    else if (location == MemoryLocation::Device) {
        /* GPU Scale */
        // cublasScale(handle, m*n, &scale, matrix, 1);
    }

	return;
}

template <typename scalar_t>
void scaleMatrix(scalar_t scale, DistributedMatrix<scalar_t>& matrix)
{
    for (int i = 0; i < matrix.gridRows(); i++) {
        for (int j = 0; j < matrix.gridColumns(); j++) {
            if (matrix.tileIsMine(i,j))
                scaleTile(scale, matrix.getTile(i, j));
        }
    }
    
	return;
}

/* Internal Scaling Functions (cuBLAS for GPU - CBLAS for CPU) */
cublasStatus_t cublasScale(cublasHandle_t handle, int n, double alpha, double* x, int incx)
{
    return cublasDscal(handle, n, &alpha, x, incx);
}

cublasStatus_t cublasScale(cublasHandle_t handle, int n, float alpha, float* x, int incx)
{
    return cublasSscal(handle, n, &alpha, x, incx);
}


void cblasScale(int n, float alpha, float* x, int incx)
{
    cblas_sscal(n, alpha, x, incx);
}

void cblasScale(int n, double alpha, double* x, int incx)
{
    cblas_dscal(n, alpha, x, incx);
}


/* TODO: Add complex/half support */
/* Strict instatiations */
template void scaleTile<float>(float scale, Tile<float>& tile);
template void scaleTile<double>(double scale, Tile<double>& tile);

template void scaleMatrix<float>(float scale, DistributedMatrix<float>& matrix);
template void scaleMatrix<double>(double scale, DistributedMatrix<double>& matrix);
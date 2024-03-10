#include "gemm.hpp"

template class GEMM<double>;
template class GEMM<int>;

template class Matrix<double>;
template class Matrix<int>;


template <typename T>
GEMM<T>::GEMM(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C, T alpha, T beta) : alpha(alpha), beta(beta), A(A), B(B), C(C), C_reference(C) {
    /* Get Problem Dimensions*/
    m = A.getRows();
    n = B.getColumns();
    k = B.getRows();

    /* Validate sizes of matrices */
    if (m != k) {
        throw std::invalid_argument("Matrix dimensions are not suitable for GEMM.");
    }

    std::cout << A << B << C << std::endl;
}

template <typename T>
void GEMM<T>::validateGEMM() {
    Matrix<T> C_reference(m,n);

    // Reference implementation (naive matrix multiplication)
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            C_reference[i * n + j] = 0.0;
            for (int l = 0; l < k; ++l) {
                C_reference[i * n + j] += alpha * A[i * k + l] * B[l * n + j];
            }
            C_reference[i * n + j] += beta * C[i * n + j];
        }
    }

    if (C_reference == C)
        std::cout << "GEMM Validation passed" << std::endl;
    else
        std::cout << "GEMM Validation failed" << std::endl;

    return;
}
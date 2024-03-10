#ifndef GEMM_HPP
#define GEMM_HPP

#include <iostream>
#include "matrix.hpp"

// C <- alpha*op(A)*op(B) + beta*C 
/* Dimensions: 
   A: m x k
   B: k x n
   C: m x n
*/
template <typename T>
class GEMM {
    private:
        T alpha, beta;
        int m,n,k;
        Matrix<T> &A, &B, &C;
        Matrix<T> C_reference;

    public:
        GEMM(Matrix<T> &A, Matrix<T> &B, Matrix<T> &C, T alpha, T beta);
        void validateGEMM();
};

#endif
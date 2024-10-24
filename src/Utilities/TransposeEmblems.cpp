#include "GeneralUtilities.hpp"

char cblasTransOpToChar(CBLAS_TRANSPOSE operation)
{
    switch(operation) {
        case CblasNoTrans:
            return 'n';
        case CblasTrans:
            return 't';
        case CblasConjTrans:
            return 'c';
        default:
            return 'n';
    }   
}

CBLAS_TRANSPOSE charToCblasTransOp(char operation)
{
    switch(operation) {
        case 'n':
        case 'N':
            return CblasNoTrans;
        case 't':
        case 'T':
            return CblasTrans;
        case 'c':
        case 'C':
            return CblasConjTrans;
        default:
            std::cerr << "Error: Invalid Transpose Operation Character: " << operation << ". Returning 'N' Transpose" << std::endl;
            return CblasNoTrans;
    }
}

char cublasTransOpToChar(cublasOperation_t operation)
{
    switch(operation) {
        case CUBLAS_OP_N:
            return 'n';
        case CUBLAS_OP_T:
            return 't';
        case CUBLAS_OP_C:
            return 'c';
        default:
            return 'n';
    } 
}

cublasOperation_t charToCublasTransOp(char operation)
{
    switch(operation) {
        case 'n':
        case 'N':
            return CUBLAS_OP_N;
        case 't':
        case 'T':
            return CUBLAS_OP_T;
        case 'c':
        case 'C':
            return CUBLAS_OP_C;
        default:
            std::cerr << "Error: Invalid Transpose Operation Character: " << operation << ". Returning 'N' Transpose" << std::endl;
            return CUBLAS_OP_N;
    }
}
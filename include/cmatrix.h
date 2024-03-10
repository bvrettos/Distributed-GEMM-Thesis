#include <iostream>

void printMatrix(double *array, int rows, int columns, int rank) {
    std::cout << "RANK: " << rank << std::endl;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            std::cout << array[i*columns + j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void generateMatrix(double *array, int rows, int columns) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            array[i*columns + j] = (double) rand() / rand();
        }
    }
}

void printInLine(double *array, int rows, int columns, int rank) {
    std::cout << "RANK: " << rank << std::endl;
    for (int i = 0; i < rows*columns; i++)
        std::cout << array[i] << " ";
    std::cout << std::endl;
}

double* copyMatrix(double *matrix1, int rows, int columns) {
    double *copiedMatrix = (double *) malloc(sizeof(double)*rows*columns);
    for (int i = 0; i < rows*columns; i++) {
        copiedMatrix[i] = matrix1[i];
    }
    return copiedMatrix;
}
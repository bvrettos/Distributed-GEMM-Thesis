#ifndef CMATRIX_H
#define CMATRIX_H

#include <iostream>
#include <vector>
#include <algorithm>

template <typename T>
bool vectorContains(std::vector<T> &vector, T value);

void printMatrix(double *array, int rows, int columns, int rank);
void generateMatrix(double *array, const int rows, const int columns);
void printInLine(double *array, int rows, int columns, int rank);
double* copyMatrix(double *matrix1, int rows, int columns);

#endif
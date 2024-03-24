#ifndef CMATRIX_H
#define CMATRIX_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <set>
#include <string>
#include <fstream>
#include <tuple>
#include <map>

template <typename T>
bool vectorContains(std::vector<T> &vector, T value);

bool findInMap(std::map<int,int> &map, int value, int* index);
void printMatrix(double *array, int rows, int columns, int rank);
void generateMatrix(double *array, const int rows, const int columns);
void printInLine(double *array, int rows, int columns, int rank);
double* copyMatrix(double *matrix1, int rows, int columns);
void writeMatrixToFile(double* matrix, const int rows, const int columns, const std::string& filename);

void writeMatrixToFileColumnMajor(double* matrix, const int rows, const int columns, const std::string& filename);
void printMatrixColumnMajor(double *array, const int rows, const int columns, int rank);
void generateMatrixColumnMajor(double *array, const int rows, const int columns);

#endif
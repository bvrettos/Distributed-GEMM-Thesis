#include "cmatrix.h"

template <typename T>
bool vectorContains(std::vector<T> &vector, T value) {
    if(std::find(vector.begin(), vector.end(), value) != vector.end())
        return true;
    else
        return false;
}

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

void generateMatrix(double *array, const int rows, const int columns) {
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

void writeMatrixToFile(double* array, const int rows, const int columns, const std::string& filename)
{
    std::ofstream outFile(filename, std::ofstream::out | std::ofstream::app);

    if (!outFile.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            outFile << array[i*columns + j] << " ";
        }
        outFile << std::endl;
    }
    outFile << std::endl;

    outFile.close();
}

bool findInMap(std::map<int,int> &map, int value, int* index)
{
    for (auto it = map.begin(); it != map.end(); ++it)
    if (it->second == value) {
        *index = it->first;
        return true;
    }

    return false;
}

void printMatrixColumnMajor(double *array, const int rows, const int columns, const int rank)
{
    printf("RANK: %d\n", rank);
    for (int i = 0; i < rows; i++) {
        for (int j = 0 ; j < columns; j++) {
            printf("%lf ", array[j*rows + i]);
        }
        printf("\n");
    }
}

void generateMatrixColumnMajor(double *array, const int rows, const int columns) 
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            array[j*rows + i] = (double) rand() / rand();
        }
    }
}

void writeMatrixToFileColumnMajor(double* array, const int rows, const int columns, const std::string& filename)
{
    std::ofstream outFile(filename, std::ofstream::out | std::ofstream::app);

    if (!outFile.is_open()) {
            std::cerr << "Error opening file: " << filename << std::endl;
            return;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            outFile << array[j*rows + i] << " ";
        }
        outFile << std::endl;
    }
    outFile << std::endl;

    outFile.close();
}

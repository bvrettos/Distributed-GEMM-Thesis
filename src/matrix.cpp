#include "matrix.hpp"

template class Matrix<double>;
template class Matrix<int>;

template std::ostream& operator<< (std::ostream& os, const Matrix<double>& matrix);
template std::ostream& operator<< (std::ostream& os, const Matrix<int>& matrix);

template Matrix<double> generateRandomMatrix(const int rows, const int columns);
template Matrix<int> generateRandomMatrix(const int rows, const int columns);


template <typename T>
Matrix<T>::Matrix(const int &rows, const int &columns) : rows(rows), columns(columns) {
    this->data = (T*) malloc(sizeof(T)*rows*columns);
    for (int i = 0; i < rows*columns; i++) {
        this->data[i] = 0;
    }
}

template <typename T>
Matrix<T>::Matrix(const int &rows, const int &columns, T initValue) : rows(rows), columns(columns) {
    this->data = (T*) malloc(sizeof(T)*rows*columns);
    for (int i = 0; i < rows*columns; i++) {
        this->data[i] = initValue;
    }
}

/* This is wrong... Need to add a check if dataArray is same size as rows*columns*/
template <typename T>
Matrix<T>::Matrix(const int &rows, const int &columns, T *data) : rows(rows), columns(columns) {    
    this->data = (T*) malloc(sizeof(T)*rows*columns);
    memcpy(this->data, data, sizeof(T)*rows*columns);
}

template <typename T>
Matrix<T>::Matrix(const int &rows, const int &columns, T **data) : rows(rows), columns(columns) {
    
}

template <typename T>
Matrix<T>::Matrix(const int &rows, const int &columns, std::vector<T>& data) : rows(rows), columns(columns) {
    /* Check size of vector */
    uint vectorSize = data.size();

    /* If number of values == rows*cols then proceed or else throw error */
    if (vectorSize != (rows*columns)) {
        std::cout << "Vector and rows/columns do not match" << std::endl;
        return;
    }

    std::cout << "Vector size and rows/columns match" << std::endl;

    /* Allocate space on Matrix->data */
    this->data = (T*) malloc(sizeof(T)*rows*columns);

    /* Copy contents of vector on row_major order */
    for (int i = 0; i < rows*columns; i++) {
        this->data[i] = data[i];
    }
}

template <typename T>
Matrix<T>::Matrix(const Matrix<T> &otherMatrix) : rows(otherMatrix.rows), columns(otherMatrix.columns) {
    data = (T*) malloc(sizeof(T)*rows*columns);
    memcpy(this->data, otherMatrix.data, sizeof(T)*rows*columns);
}

template <typename U>
std::ostream& operator << (std::ostream& os, const Matrix<U>& matrix) {
    std::cout << matrix.rows << " " << matrix.columns << std::endl;
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0;j < matrix.columns; j++) {
            os << matrix.data[matrix.columns*i + j] << ' ';
        }
        os << std::endl;
    }
    return os;
}

template <typename T>
T& Matrix<T>::operator[] (int index) {
    if (index < 0 || index >= rows*columns) {
        throw std::out_of_range("Index out of bounds");
    }
    return data[index];
}

template <typename T>
const T& Matrix<T>::operator[] (int index) const {
    if (index < 0 || index >= rows*columns) {
        throw std::out_of_range("Index out of bounds");
    }
    return data[index];
}

template<typename T>
Matrix<T>::~Matrix() {
    // free(this->data);
}

template <typename T>
bool Matrix<T>::operator == (const Matrix<T>& otherMatrix) const {
    if ((rows != otherMatrix.rows) || (columns != otherMatrix.columns)) {
        return false;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            if (data[i*columns + j] != otherMatrix[i*columns + j])
                return false;
        }
    }
    return true;
}

template <typename T>
const Matrix<T>& Matrix<T>::operator = (const Matrix<T> &otherMatrix) {
    rows = otherMatrix.rows;
    columns = otherMatrix.columns;

    data = (T*) malloc(sizeof(T)*rows*columns);

    memcpy(this->data, otherMatrix.data, sizeof(T)*rows*columns);

    return *this;
}

template <typename T>
bool Matrix<T>::isSquare()
{
    return rows == columns;
}

template <typename T>
T* Matrix<T>::getDataPointer() {
    return data;
}

template <typename T>
const T* Matrix<T>::getDataPointer() const {
    return data;
}

template <typename T>
void Matrix<T>::writeMatrix(const std::string& fileName) const {
    std::ofstream outFile(fileName);

    if (!outFile.is_open()) {
            std::cerr << "Error opening file: " << fileName << std::endl;
            return;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            outFile << data[i*columns + j] << " ";
        }
        outFile << std::endl;
    }
    outFile.close();
}

template <typename T>
int Matrix<T>::getRows() const {
    return rows;
}

template <typename T>
int Matrix<T>::getColumns() const {
    return columns;
}

template <typename T>
Matrix<T> generateRandomMatrix(const int rows, const int columns)
{
    Matrix<T> matrix(rows, columns);
    for (int i = 0; i < rows*columns; i++) {
        matrix[i] = (T) rand() / rand();
    }

    return matrix;
}
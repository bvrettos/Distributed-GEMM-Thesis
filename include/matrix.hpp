#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>

template <typename T>
class Matrix {
    private:
        int rows, columns;
        T* data;
        bool empty;

    public:
        /* Constructors */
        Matrix();
        Matrix(const int &rows, const int &columns);
        Matrix(const int &rows, const int &columns, T initValue);
        Matrix(const int &rows, const int &columns, T *data);
        Matrix(const int &rows, const int &columns, T **data);
        Matrix(const int &rows, const int &columns, std::vector<T>& data);
        Matrix(const Matrix<T> &otherMatrix);

        /* Destructor */
        ~Matrix();  

        /* Operators */
        T &operator [] (int index); //write
        const T& operator [] (int index) const; //read-only
        
        T* getDataPointer();
        const T* getDataPointer() const;

        template <typename U>
        friend std::ostream& operator << (std::ostream &os, const Matrix<U>& matrix);

        bool operator == (const Matrix<T>& otherMatrix) const;
        const Matrix<T>& operator = (const Matrix<T> &otherMatrix);
        
        /* Helper Member functions*/
        bool isSquare();
        void writeMatrix(const std::string& fileName) const;
        int getRows() const;
        int getColumns() const;
};



template <typename T>
Matrix<T> generateRandomMatrix(const int rows, const int columns);

#endif
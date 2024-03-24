#ifndef VALIDATION_HPP
#define VALIDATION_HPP

#include <cfloat>
#include <cstdio>

template <typename T>
double abs(T x);

template <typename T>
inline T Derror(T a, T b);

template <typename T>
long int vectorDifference(T* a, T* b, long long size, double eps);

template <typename T>
short testEquality();

/* Petyros */
short Dtest_equality(double* C_comp, double* C, long long size);
long int Dvec_diff(double* a, double* b, long long size, double eps);
inline double Derror(double a, double b);
double dabs(double x);

#endif
#ifndef TRANSFERS_HPP
#define TRANSFERS_HPP

#include <mpi.h>
#include <typeinfo>
#include <errorHandling.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include <cuda.h>
#include <cuda_runtime.h>

extern std::map<std::tuple<long, long, long>, MPI_Datatype*> datatypeCache;
extern bool cacheInitialized;

template <typename scalar_t>
void transferBlock(long long rows, long long columns, scalar_t* sourcePointer, const long long stride, const int destinationRank, int tag, MPI_Request* requestHandle, bool colMajor=true);

template <typename scalar_t>
void receiveBlock(long long rows, long long columns, scalar_t* destinationPointer, const long long stride, const int sourceRank, int tag, MPI_Request* requestHandle, bool colMajor=true);

bool findDatatypeInCache(long long rows, long long columns, long long stride);

template <typename T>
void copy(size_t size, const T* sourcePointer, T* destinationPointer, cudaMemcpyKind cudaCopyDirection);

template <typename T>
void copyBlock(long long rows, long long columns, T* sourcePointer, T* destinationPointer, const long long ldSource, const long long ldDestination, bool columnMajor=true);

#endif
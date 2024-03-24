# CMake version requirement
cmake_minimum_required(VERSION 3.10)

# Project name
project(MPI-CUDA)

# Set C++ standard (optional)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA Stuff
set(CUDA_PREFIX "/usr/local/cuda-12.1")
set (CUDA_LD ${CUDA_PREFIX}/lib64) # or /lib depending on system
set (CUDA_LINK "-L${CUDA_LD} -lcublas -lcurand -lcuda -lcudart")
set (CUDA_INCLUDE_DIRS ${CUDA_PREFIX}/include ${CUDA_PREFIX}/samples/common/inc)
set (CMAKE_CUDA_COMPILER ${CUDA_PREFIX}/bin/nvcc)

# Set MPI Stuff
set (MPI_PREFIX $ENV{OMPI_PATH})
set (MPI_INCLUDE_DIRS ${MPI_PREFIX}/include)
set (MPI_LD ${MPI_PREFIX}/lib)
set (MPI_LINK "-lmpi -L${MPI_LD}")

# Set PARALiA Stuff
set (PARALIA_PREFIX "/home/users/vvrettos/PARALiA-GEMMex/build/silver1_2V100-install")
set (PARALIA_LINK "-L${PARALIA_PREFIX}/lib -lparalia -lautotuner -lcaching -lchl_grid_amalgamation -lchl_smart_wrappers")
set (PARALIA_INCLUDE_DIRS ${PARALIA_PREFIX}/include)

set(CMAKE_C_COMPILER ${CMAKE_CUDA_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_CUDA_COMPILER})

# Set OpenBLAS Stuff
set (OPENBLAS_PREFIX $ENV{OPENBLAS_PATH})
set (OPENBLAS_LD ${OPENBLAS_PREFIX}/lib)
set (OPENBLAS_LINK "-L${OPENBLAS_LD} -lopenblas")
set (OPENBLAS_INCLUDE_DIRS ${OPENBLAS_PREFIX}/include)

# Set cuBLASMp Stuff
set (CUBLASMP_PREFIX $ENV{CUBLASMP_PATH})
set (CUBLASMP_INCLUDE_DIRS ${CUBLASMP_PREFIX}/include)
set (CUBLASMP_LD ${CUBLASMP_PREFIX}/lib)
set (CUBLASMP_LINK "-L${CUBLASMP_LD} -lcublasmp")

set (CAL_PREFIX $ENV{CAL_PATH})
set (CAL_INCLUDE_DIRS ${CAL_PREFIX}/include)
set (CAL_LD ${CAL_PREFIX}/lib)
set (CAL_LINK "-L${CAL_LD} -lcal")

# Set numactl Stuff
set (NUMACTL_PREFIX $ENV{NUMACTL_PATH})
set (NUMACTL_INCLUDE_DIRS ${NUMACTL_PREFIX}/include)
set (NUMACTL_LINK "-L${NUMACTL_PATH}/lib -lnuma")

# Set SCALAPACK Stuff
set (SCALAPACK_PREFIX $ENV{SCALAPACK_PATH})
set (SCALAPACK_INCLUDE_DIRS ${SCALAPACK_PREFIX}/include)
set (SCALAPACK_LINK "-L${SCALAPACK_PREFIX}/lib -lscalapack")

add_definitions(-DMAX_BACKEND_L=8)
add_definitions(-DREDUCE_WORKERS_PERDEV=10)
add_definitions(-DPROBLEM_MD_CACHE=100) # Only relevant for "PROBLEMS" reuse level

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${PARALIA_INCLUDE_DIRS}
    ${MPI_INCLUDE_DIRS}
    ${OPENBLAS_INCLUDE_DIRS}
    ${CUBLASMP_INCLUDE_DIRS}
    ${CAL_INCLUDE_DIRS}
    ${NUMACTL_INCLUDE_DIRS}
    ${SCALAPACK_INCLUDE_DIRS}
    include
)

add_executable(cublasmpTest
    src/cuBLASMPtest.cpp src/cmatrix.cpp src/pblasDecomposition.cpp
    src/validation.cpp
)

add_executable(mpiBenchmark
    src/mpiBenchmark.cpp src/cudaCommunicator.cpp src/mpiBLAS.cpp 
    src/2DBlockCyclicDecomposition.cpp src/2DBlockSequentialDecomposition.cpp src/validation.cpp
    src/cmatrix.cpp
)

add_executable(pblasGEMM
    src/pblasGEMM.cpp src/pblasDecomposition.cpp
    src/cmatrix.cpp src/validation.cpp
)

target_link_libraries(cublasmpTest ${MPI_LINK} ${OPENBLAS_LINK} ${CAL_LINK} ${CUBLASMP_LINK} ${CUDA_LINK} ${SCALAPACK_LINK})
target_link_libraries(mpiBenchmark ${MPI_LINK} ${PARALIA_LINK} ${OPENBLAS_LINK} ${NUMACTL_LINK} ${CUDA_LINK})
target_link_libraries(pblasGEMM ${MPI_LINK} ${OPENBLAS_LINK} ${SCALAPACK_LINK})
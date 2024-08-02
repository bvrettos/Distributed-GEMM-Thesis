# CMake version requirement
cmake_minimum_required(VERSION 3.10)

# Project name
project(MPI-CUDA LANGUAGES C CXX CUDA)

# Set C++ standard (optional)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Set CUDA Stuff
set (CUDA_PREFIX $ENV{CUDA_PATH})
set (CUDA_LD ${CUDA_PREFIX}/lib64) # or /lib depending on system
set (CUDA_LINK "-L${CUDA_LD} -lcublas -lcurand -lcuda -lcudart -lcusolver -lrt")
set (CUDA_INCLUDE_DIRS ${CUDA_PREFIX}/include ${CUDA_PREFIX}/samples/common/inc)
set (CMAKE_CUDA_COMPILER ${CUDA_PREFIX}/bin/nvcc)
set(CMAKE_C_COMPILER ${CMAKE_CUDA_COMPILER})
set(CMAKE_CXX_COMPILER ${CMAKE_CUDA_COMPILER})

# Set MPI Stuff
set (MPI_PREFIX $ENV{OPENMPI_PATH})
set (MPI_INCLUDE_DIRS ${MPI_PREFIX}/include)
set (MPI_LD ${MPI_PREFIX}/lib)
set (MPI_LINK "-lmpi -L${MPI_LD}")

# Set MPI Stuff
set (NVHPC_MPI_PREFIX $ENV{EBROOTOPENMPI})
set (NVHPC_MPI_INCLUDE_DIRS ${NVHPC_MPI_PREFIX}/include)
set (NVHPC_MPI_LD ${NVHPC_MPI_PREFIX}/lib)
set (NVHPC_MPI_LINK "-lmpi -L${MPI_LD}")

# Set PARALiA Stuff
set (PARALIA_PREFIX $ENV{PARALIA_PATH})
set (PARALIA_LINK "-L${PARALIA_PREFIX}/lib -lparalia -lautotuner -lcaching -lchl_grid_amalgamation -lchl_smart_wrappers")
set (PARALIA_INCLUDE_DIRS ${PARALIA_PREFIX}/include)

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

# Set SLATE Stuff
set (SLATE_PREFIX $ENV{SLATE_PATH})
set (SLATE_INCLUDE_DIRS ${SLATE_PREFIX}/include)
set (SLATE_LINK "-L${SLATE_PREFIX}/lib64 -lslate -lomp -llapackpp -lslate_lapack_api -lblaspp")

# Set COSMA Stuff
set (COSMA_PREFIX $ENV{COSMA_PATH})
set (COSMA_INCLUDE_DIRS ${COSMA_PREFIX}/include)
set (COSMA_LINK "-L${COSMA_PREFIX}/lib64 -lTiled-MM -lcosma -lcosta -lgomp")

set (NCCL_PREFIX $ENV{NCCL_PATH})
set (NCCL_INCLUDE_DIRS ${NCCL_PREFIX}/include)
set (NCCL_LINK "-L{NCCL_PREFIX}/lib -lnccl")

################################################################################################
#---------------------------Settings and Tunable Parameters------------------------------------#
################################################################################################
set(MAX_BACKEND_L 8)
add_definitions(-DMAX_BACKEND_L_IN=${MAX_BACKEND_L})
add_definitions(-DDTYPE_NUM=3)
add_definitions(-DREDUCE_WORKERS_PERDEV=10)
add_definitions(-DPROBLEM_MD_CACHE=100) # Only relevant for "PROBLEMS" reuse level
add_definitions(-DMACHINE_NAME="meluxina")

include_directories(
    ${CUDA_INCLUDE_DIRS}
    ${PARALIA_INCLUDE_DIRS}
    ${MPI_INCLUDE_DIRS}
    ${OPENBLAS_INCLUDE_DIRS}
    ${CUBLASMP_INCLUDE_DIRS}
    ${CAL_INCLUDE_DIRS}
    ${NUMACTL_INCLUDE_DIRS}
    ${SCALAPACK_INCLUDE_DIRS}
    ${SLATE_INCLUDE_DIRS}
    ${COSMA_INCLUDE_DIRS}
    # ${NVHPC_MPI_INCLUDE_DIRS}
    include
)

#######################################################################################
#--------------------------- WorkInProgress Executables ------------------------------#
#######################################################################################

# add_executable(pblasGEMM
#     src/pblasGEMM.cpp src/pblasDecomposition.cpp
#     src/cmatrix.cpp src/validation.cpp src/cudaCommunicator.cpp src/generalUtilities.cpp
# )

add_executable(sequentialDecomposerTest
    src/sequentialDecomposerTest.cpp src/2DBlockSequentialDecomposition.cpp src/cmatrix.cpp
    src/transfers.cpp  src/generalUtilities.cpp
)

add_executable(cudaAwareMpi
    src/cudaAwareMpi.cpp
)

add_executable(summa25D
    src/25DSumma.cpp src/summa25Dtest.cpp src/cmatrix.cpp src/transfers.cpp src/generalUtilities.cpp src/validation.cpp
    src/logging.cpp
)

add_executable(cublas25DBenchmark
    src/25DSumma.cpp src/cuBLAS25Benchmark.cpp src/cmatrix.cpp src/transfers.cpp src/generalUtilities.cpp src/validation.cpp
    src/logging.cpp
)

# target_link_libraries(pblasGEMM ${MPI_LINK} ${OPENBLAS_LINK} ${SCALAPACK_LINK} ${CUDA_LINK})
target_link_libraries(sequentialDecomposerTest ${MPI_LINK} ${CUDA_LINK})
target_link_libraries(cudaAwareMpi ${MPI_LINK} ${CUDA_LINK})
target_link_libraries(summa25D ${MPI_LINK} ${CUDA_LINK} ${OPENBLAS_LINK})
target_link_libraries(cublas25DBenchmark ${MPI_LINK} ${CUDA_LINK} ${OPENBLAS_LINK})

# #######################################################################################
# #--------------------------- PARALiA Executables ------------------------------------#
# #######################################################################################

# add_executable(singleNodeParaliaBenchmark
#     src/singleNodeParaliaBenchmark.cpp src/paraliaWrappers.cpp src/cmatrix.cpp src/logging.cpp
#     src/generalUtilities.cpp
# )

# add_executable(distributedParaliaBenchmark
#     src/distributedParaliaBenchmark.cpp src/distributedParalia.cpp src/cmatrix.cpp src/logging.cpp
#     src/generalUtilities.cpp src/2DBlockSequentialDecomposition.cpp src/transfers.cpp
# )

# add_executable(distributedParaliaValidators
#     src/distributedParaliaValidators.cpp src/distributedParalia.cpp src/cmatrix.cpp src/logging.cpp
#     src/generalUtilities.cpp src/2DBlockSequentialDecomposition.cpp src/transfers.cpp
# )

# target_link_libraries(singleNodeParaliaBenchmark ${PARALIA_LINK} ${CUDA_LINK} ${NUMACTL_LINK})
# target_link_libraries(distributedParaliaBenchmark ${PARALIA_LINK} ${MPI_LINK} ${CUDA_LINK} ${NUMACTL_LINK} ${OPENBLAS_LINK})
# target_link_libraries(distributedParaliaValidators ${PARALIA_LINK} ${MPI_LINK} ${CUDA_LINK} ${NUMACTL_LINK} ${OPENBLAS_LINK})


# #######################################################################################
# #--------------------------- cuBLASMp Executables ------------------------------------#
# #######################################################################################
# add_executable(cublasmpBenchmark
#     src/cuBLASMpBenchmark.cpp src/cuBLASMpWrapped.cpp 
#     src/pblasDecomposition.cpp src/cudaCommunicator.cpp src/calSetup.cpp
#     src/cmatrix.cpp src/logging.cpp src/generalUtilities.cpp src/transfers.cpp
# )

# add_executable(cublasmpValidator
#     src/cuBLASMpValidator.cpp src/cuBLASMpWrapped.cpp 
#     src/pblasDecomposition.cpp src/cudaCommunicator.cpp src/calSetup.cpp
#     src/validation.cpp src/cmatrix.cpp src/logging.cpp src/generalUtilities.cpp
#     src/transfers.cpp
# )

# target_link_libraries(cublasmpBenchmark ${MPI_LINK} ${OPENBLAS_LINK} ${CAL_LINK} ${CUBLASMP_LINK} ${CUDA_LINK} ${SCALAPACK_LINK})
# target_link_libraries(cublasmpValidator ${MPI_LINK} ${OPENBLAS_LINK} ${CAL_LINK} ${CUBLASMP_LINK} ${CUDA_LINK} ${SCALAPACK_LINK})

# ##################################################################################
# #---------------------------SLATE Executables------------------------------------#
# ##################################################################################
# add_executable(slateBenchmark
#     src/slateBenchmark.cpp src/validation.cpp src/cmatrix.cpp src/cudaCommunicator.cpp src/slateWrapped.cpp
#     src/logging.cpp src/generalUtilities.cpp
# )

# add_executable(slateValidator
#     src/slateValidator.cpp src/validation.cpp src/cmatrix.cpp src/cudaCommunicator.cpp src/slateWrapped.cpp
#     src/logging.cpp src/generalUtilities.cpp
# )
# target_link_libraries(slateValidator ${MPI_LINK} ${OPENBLAS_LINK} ${SLATE_LINK} ${CUDA_LINK} ${SCALAPACK_LINK})
# target_link_libraries(slateBenchmark ${MPI_LINK} ${OPENBLAS_LINK} ${SLATE_LINK} ${CUDA_LINK} ${SCALAPACK_LINK})

# ##################################################################################
# #---------------------------COSMA Executables------------------------------------#
# ##################################################################################
# add_executable(cosmaBenchmark
#     src/cosmaBenchmark.cpp src/validation.cpp src/cmatrix.cpp src/cosmaWrapped.cpp src/generalUtilities.cpp
#     src/validation.cpp src/cudaCommunicator.cpp src/logging.cpp src/pblasDecomposition.cpp src/transfers.cpp
# )

# add_executable(cosmaValidators
#     src/cosmaValidators.cpp src/validation.cpp src/cmatrix.cpp src/cosmaWrapped.cpp src/generalUtilities.cpp
#     src/validation.cpp src/cudaCommunicator.cpp src/logging.cpp src/pblasDecomposition.cpp src/transfers.cpp
# )

# target_link_libraries(cosmaBenchmark ${MPI_LINK} ${COSMA_LINK} ${CUDA_LINK} ${SCALAPACK_LINK} ${NCCL_LINK} ${OPENBLAS_LINK})
# target_link_libraries(cosmaValidators ${MPI_LINK} ${COSMA_LINK} ${CUDA_LINK} ${SCALAPACK_LINK} ${NCCL_LINK} ${OPENBLAS_LINK})
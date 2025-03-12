# Optimization of distributed dense matrix multiplication algorithms for optimal usage in modern GPU clusters

This work focuses on the study of state-of-practice distributed algorithms for general matrix multiplication and their shortcomings when executed in modern multi-node multi-GPU systems. By utilizing special features of contemporary GPUs, we present our own method of execution, built upon existing algorithms, outperforming other Parallel Basic Linear Algebra Subprogram (PBLAS) libraries.

## Evaluation 

The performance evaluation stage was conducted on [MeluXina](https://www.luxprovide.lu/meluxina/) by comparing our proposed systems with:

- [cuBLASMp](https://docs.nvidia.com/cuda/cublasmp/index.html), a PBLAS library specifically designed for matrix multiplication.
- [SLATE](https://github.com/icl-utk-edu/slate), the successor of ScaLAPACK optimized for modern GPU-accelerated systems.
- [COSMA](https://github.com/eth-cscs/COSMA), a novel communication optimal system using the red-blue pebble game algorithm.

Our systems outperform state-of-the-art libraries on all benchmark scenarios by a factor of up to 3.3x compared to the next best performing system.

## Abstract

Multiplication of Dense Matrices is one of the most common and important mathematical kernels
executed in both normal systems and clusters alike. Following the major increase of GPUs in the
HPC world due to the recent ”AI boom”, the GEMM kernel is commonly found being executed
by Graphics Processors instead of the conventional CPUs. The previous distributed algorithms
used by the GEMM calls were heavily designed for use in CPU based systems and fall short when
executed in modern GPU nodes. In this thesis, we study these distributed algorithms as well as
PBLAS libraries that focus on GPU execution in an attempt to optimize them for use in modern
cluster systems. In the process, we present our own method of execution, built upon existing
algorithms, modified in such a way that many of the special features of contemporary Graphics
Processors are utilized in order to increase performance.

## Keywords

Distributed Dense Matrix Multiplication, Graphics Processors (GPUs), Cluster Systems, HPC, Distributed Algorithm Optimization, Communication/Computation Overlap


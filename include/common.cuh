#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(x) do {\
    cudaError_t e = (x);\
    if (e != cudaSuccess) {\
        fprintf(stderr, "CUDA error: %s:%d, %s \n",\
            __FILE__, __LINE__, cudaGetErrorString(e));\
        exit(1);\
    }\
} while(0)

/**
 * calculate a/b with round up instead of truncating.
 * ensure that the grid covers the whole matrix when launching
 */
#define CEIL_DIV(a, b) (((a) + (b) - 1) / b)
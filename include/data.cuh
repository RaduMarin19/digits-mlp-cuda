#pragma once

#include <random>
#include <algorithm>

#include "common.cuh"

typedef struct{
    float *X; // [n, d] on device
    int *y; // [n] on device
    int n;
    int d;
    int C;
} Dataset;

void dataset_load(Dataset* ds, const char* path);
void dataset_free(Dataset* ds);

void shuffle(int* idx, int n);

__global__ void gather_rows_kernel(const float* X, 
                                const int* idx, 
                                float* X_batch, 
                                int d, 
                                int batch);

void gather_rows(const float* X, 
                const int* idx, 
                float* X_batch, 
                int d, 
                int batch);

__global__ void gather_labels_kernel(const int* y,
                                    const int* idx,
                                    int* y_batch,
                                    int batch);

void gather_labels(const int* y,
                const int* idx,
                int* y_batch,
                int batch);

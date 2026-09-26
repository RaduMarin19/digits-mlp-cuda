#pragma once

typedef struct{
    float *X; // [n, d] on device
    int *y; // [n] on device
    int n;
    int d;
    int C;
} Dataset;

void dataset_load(Dataset* ds, const char* path);
void dataset_free(Dataset* ds);

void gather_rows(const float* X, const int* idx, float* X_batch, int d, int batch);
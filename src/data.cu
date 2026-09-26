#include "data.cuh"
#include "common.cuh"
#include <cstdio>
#include <cstdlib>

void dataset_load(Dataset* ds, const char* path){
    FILE* f = fopen(path, "rb");
    if (!f) {fprintf(stderr, "cannot open %s/n", path); exit(1);}

    int hdr[3];
    fread(hdr, sizeof(int), 3, f);
    ds->n = hdr[0];
    ds->d = hdr[1];
    ds->C = hdr[2];

    float *hX = (float*)malloc((size_t)ds->n * ds->d * sizeof(float));
    int *hY = (int*)malloc((size_t)ds->n * sizeof(int));
    fread(hX, sizeof(float), (size_t)ds->n*ds->d, f);
    fread(hY, sizeof(int), (size_t)ds->n, f);
    fclose(f);

    CUDA_CHECK(cudaMalloc(&ds->X, (size_t)ds->n * ds->d * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ds->y, (size_t)ds->n * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(ds->X, hX, (size_t)ds->n * ds->d * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ds->y, hY, (size_t)ds->n * sizeof(int), cudaMemcpyHostToDevice));

    free(hX);
    free(hY);
}

void dataset_free(Dataset* ds){
    cudaFree(ds->X);
    cudaFree(ds->y);
}

void gather_rows(const float* X, const int* idx, float* X_batch, int d, int batch){
    int row = blockIdx.x;
    int col = threadIdx.x;
    if (row >= batch || col >= d) return;
    X_batch[row * d + col] = X[idx[row] * d + col];
}
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

void shuffle(int* idx, int n) {
    std::mt19937 rng{std::random_device{}()};

    for (int i = n - 1; i > 0; --i) {
        std::uniform_int_distribution<int> dist(0, i);
        std::swap(idx[i], idx[dist(rng)]);
    }
}

__global__ void gather_rows_kernel(const float* X, const int* idx, float* X_batch, int d, int batch) {
    for (int row = blockIdx.x; row < batch; row += gridDim.x)
        for (int col = threadIdx.x; col < d; col += blockDim.x)
            X_batch[row * d + col] = X[idx[row] * d + col];
}

void gather_rows(const float* X, const int* idx, float* X_batch, int d, int batch){
    dim3 threads(256);
    dim3 grid(256);
    gather_rows_kernel<<<grid, threads>>>(X, idx, X_batch, d, batch);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void gather_labels_kernel(const int* y, const int* idx,
                                     int* y_batch, int batch) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batch; i += gridDim.x * blockDim.x)
        y_batch[i] = y[idx[i]];
}

void gather_labels(const int* y, const int* idx, int* y_batch, int batch) {
    gather_labels_kernel<<<256, 256>>>(y, idx, y_batch, batch);
    CUDA_CHECK(cudaGetLastError());
}
// tests/test_layers.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include "common.cuh"
#include "layers.cuh"
#include "test_utils.cuh"

static void cpu_softmax_xent(const float* Z, const int* y, float* probs, float* loss, int M, int N){
    for(int row = 0; row < M; ++row){
        float max = Z[row * N];
        for(int i = 1; i < N; ++i){
            if(max < Z[row * N + i])
                max = Z[row * N + i];
        }
        double sum = 0;
        for(int i = 0; i < N; ++i){
            probs[row * N + i] = exp(Z[row * N + i] - max);
            sum += probs[row * N + i];
        }
        for(int i = 0; i < N; ++i){
            probs[row * N + i] /= sum;
        }
        loss[row] = -log(probs[row * N + y[row]]);
    }
}

static int check(int M, int N){
    size_t nZ = (size_t) M * N, nY = (size_t) M, nP = (size_t) M * N, nL = (size_t) M;
    float* hZ = (float*)malloc(nZ * sizeof(float));
    int* hY = (int*)malloc(nY * sizeof(int));
    float* hP = (float*)malloc(nP * sizeof(float));
    float* hL = (float*)malloc(nL * sizeof(float));
    float* refP = (float*)malloc(nP * sizeof(float));
    float* refL = (float*)malloc(nL * sizeof(float));

    unsigned seed = 12345;
    fill(hZ, nZ, seed);
    fill_labels(hY, nY, N, seed);
    cpu_softmax_xent(hZ, hY, refP, refL, M, N);

    float* dZ, *dP, *dL;
    int* dY;
    CUDA_CHECK(cudaMalloc(&dZ, nZ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dY, nY * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dP, nP * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dL, nL * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dZ, hZ, nZ * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dY, hY, nY * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dP, 0xFF, nP * sizeof(float)));
    CUDA_CHECK(cudaMemset(dL, 0xFF, nL * sizeof(float)));
    memset(hP, 0xFF, nP * sizeof(float));
    memset(hL, 0xFF, nL * sizeof(float));
    
    softmax_xent(dZ, dY, dP, dL, M, N);

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hP, dP, nP * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hL, dL, nL * sizeof(float), cudaMemcpyDeviceToHost));

    int ok = close(hP, refP, nP) && close(hL, refL, nL);
    printf("%-8s M=%-4d N=%-4d \n", ok ? "[ok]" : "[FAIL]", M, N);

    cudaFree(dZ);
    cudaFree(dY);
    cudaFree(dP);
    cudaFree(dL);
    free(hZ);
    free(hY);
    free(hP);
    free(hL);
    free(refP);
    free(refL);
    return ok;
}

int main(){
    int fails = 0;

    fails += !check(100, 10);
    fails += !check(3, 101);
    fails += !check(1000, 2);
    fails += !check(17, 22);
    fails += !check(10, 250);
    fails += !check(112, 69);
    fails += !check(10, 11);
    fails += !check(420, 9);
    fails += !check(300, 720);
    printf(fails ? "\n%d FAILED\n" : "\nall passed\n", fails);
    return fails != 0;
}

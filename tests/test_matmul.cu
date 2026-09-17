// tests/test_matmul.cu
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "common.cuh"
#include "matmul.cuh"
#include "test_utils.cuh"

static void cpu_gemm(const float* A, const float* B, float* C,
                     int M, int N, int K) {
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;                    // double: reference, not speed
            for (int k = 0; k < K; ++k)
                acc += (double)A[m*K + k] * B[k*N + n];
            C[m*N + n] = (float)acc;
        }
}

static int check(int M, int N, int K) {
    size_t nA = (size_t)M*K, nB = (size_t)K*N, nC = (size_t)M*N;
    float *hA = (float*)malloc(nA*4), *hB = (float*)malloc(nB*4);
    float *hC = (float*)malloc(nC*4), *ref = (float*)malloc(nC*4);

    unsigned seed = 12345;
    fill(hA, nA, seed);
    fill(hB, nB, seed);
    cpu_gemm(hA, hB, ref, M, N, K);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, nA*4));
    CUDA_CHECK(cudaMalloc(&dB, nB*4));
    CUDA_CHECK(cudaMalloc(&dC, nC*4));
    CUDA_CHECK(cudaMemcpy(dA, hA, nA*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, nB*4, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(dC, 0xFF, nC*4));      // poison: catch unwritten cells

    gemm(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hC, dC, nC*4, cudaMemcpyDeviceToHost));

    int ok = close(hC, ref, nC);
    printf("%-8s M=%-4d N=%-4d K=%-4d\n", ok ? "[ok]" : "[FAIL]", M, N, K);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC); free(ref);
    return ok;
}

int main() {
    int fails = 0;
    fails += !check(  1,   1,   1);   // degenerate
    fails += !check(  1,  10,  64);   // single row  (batch of 1)
    fails += !check( 64,  10,   1);   // K=1
    fails += !check( 13,   7,  23);   // all prime, no dim divisible by 16
    fails += !check( 16,  16,  16);   // exactly one tile
    fails += !check( 17,  33,  31);   // one past tile boundaries
    fails += !check( 64, 128,  64);   // a real layer shape
    fails += !check(257, 129, 513);   // larger, multi-block
    printf(fails ? "\n%d FAILED\n" : "\nall passed\n", fails);
    return fails != 0;
}
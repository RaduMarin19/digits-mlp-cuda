#include "optim.cuh"

__global__ void sgd_momentum(float *__restrict__ W, float* __restrict__ v, const float *__restrict__ dW, int M, int N, float lr, float miu)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    float vi = miu * v[idx] - lr * dW[idx];
    v[idx] = vi;
    W[idx] += vi;
}
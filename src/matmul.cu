
#include "matmul.cuh"

__global__ void gemm_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) { 
    // calculate position in matrix C for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N)
        return;

    // calculate the value and assign
    float acc = 0.f;
    for(int i = 0; i < K; ++i){
        acc += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = acc;
}

void gemm(const float *A, const float *B, float *C, int M, int N, int K)
{  
    dim3 block(BLOCK_SIZE, BLOCK_SIZE); // BLOCK_SIZE^2 threads
    dim3 grid(CEIL_DIV(N, block.x), // x cols
            CEIL_DIV(M, block.y)); // y rows
    gemm_kernel<<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}


#include "matmul.cuh"

template<bool TA, bool TB>
__global__ void gemm_kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) { 
    // calculate position in matrix C for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N)
        return;

    // calculate the value and assign
    float acc = 0.f;
    for(int i = 0; i < K; ++i){
        float a = TA ? A[i * M + row] : A[row * K + i];
        float b = TB ? B[col * K + i] : B[i * N + col];
        acc += a * b;
    }
    C[row * N + col] = acc;
}

void gemm(const float *A, const float *B, float *C, int M, int N, int K, bool ta, bool tb)
{  
    dim3 block(BLOCK_SIZE, BLOCK_SIZE); // BLOCK_SIZE^2 threads
    dim3 grid(CEIL_DIV(N, block.x), // x cols
            CEIL_DIV(M, block.y)); // y rows
    if (!ta && !tb) gemm_kernel<false, false><<<grid, block>>>(A, B, C, M, N, K);
    else if (ta && !tb) gemm_kernel<true, false><<<grid, block>>>(A, B, C, M, N, K);
    else if (!ta && tb) gemm_kernel<false, true><<<grid, block>>>(A, B, C, M, N, K);
    else gemm_kernel<true, true><<<grid, block>>>(A, B, C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
}

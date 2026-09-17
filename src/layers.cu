#include "layers.cuh"

__global__ void add_bias(float* __restrict__ Z, const float* __restrict__ b, int M, int N) { 
    // calculate position in Z
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N)
        return;
    
    Z[row * N + col] += b[col];
}

__global__ void add_bias_relu(float* __restrict__ Z, const float* __restrict__ b, int M, int N) {
    // calculate position in Z
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N)
        return;
    
    float result = Z[row * N + col];
    result += b[col];
    Z[row * N + col] = result > 0 ? result : 0;
}

__global__ void softmax_xent_kernel(const float *Z, const int *__restrict__ y, float *__restrict__ probs, float *__restrict__ loss, int M, int N)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // for softmax -> sum for row in Z -> each element in Z gets divided by the sum for its row
    // important: for softmax calculate the exponential of xi - xmax for numerical stability
    // for cross-entropy calculate with -log(prob(y_true)) and save in loss

    // compute the max m_row for each row in Z
    __shared__ float sm[SOFTMAX_BLOCK];

    // sm[tid] = Z[index]; doesn't work -> for 32 threads threads 10-31 write trash because N = 10
    // this approach works for any N. it is only partially parallel for N > threadsPerBlock.
    float m = -INFINITY;
    for (int j = tid; j < N; j += blockDim.x)
        m = fmaxf(m, Z[row * N + j]);
    sm[tid] = m;
    __syncthreads();
    for(int s = blockDim.x / 2; s > 0; s >>= 1){
        if(tid < s) sm[tid] = fmaxf(sm[tid], sm[tid + s]);
        __syncthreads();
    }
    // max is in sm[0]
    
    // probs = exp(Z[row * N + tid] - m_row)
    // float e = expf(Z[index] - sm[0]);
    // probs[index] = e;
    // __syncthreads();
    // sm[tid] = e;
    // but same principle as before for N > threadsPerBlock
    m = sm[0];
    __syncthreads();
    float partial = 0.f;
    for (int j = tid; j < N; j += blockDim.x){
        float e = expf(Z[row * N + j] - m);
        probs[row * N + j] = e;
        partial += e;
    }
    sm[tid] = partial;

    __syncthreads();
    for(int s = blockDim.x / 2; s > 0; s >>= 1){
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }

    // sum is in sm[0];
    // probs[index] = probs[index] / sm[0];
    float sum = sm[0];
    for (int j = tid; j < N; j += blockDim.x){
        probs[row * N + j] /= sum;
    }

    // y[row] is an class index (0-9);
    // multi-class cross-entropy collapses to a single lookup, picking up the probability of the correct class.
    __syncthreads();
    if(tid == 0 && loss)
        loss[row] = -logf(fmaxf(probs[row * N + y[row]], 1e-12f)); // guard agains log underflow
}

__host__ void softmax_xent(const float *Z, const int *y, float *probs, float *loss, int M, int N)
{
    dim3 block(SOFTMAX_BLOCK);
    dim3 grid(M); // x cols
    softmax_xent_kernel<<<grid, block>>>(Z, y, probs, loss, M, N);
    CUDA_CHECK(cudaGetLastError());
}

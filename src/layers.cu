#include "layers.cuh"

__global__ void add_bias_kernel(float* __restrict__ Z, const float* __restrict__ b, int M, int N) { 
    // calculate position in Z
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N)
        return;
    
    Z[row * N + col] += b[col];
}

__host__ void add_bias(float *Z, const float *b, int M, int N)
{
    dim3 block(ADDBIAS_BLOCK, ADDBIAS_BLOCK);
    dim3 grid(CEIL_DIV(N, ADDBIAS_BLOCK), 
            CEIL_DIV(M, ADDBIAS_BLOCK));
    add_bias_kernel<<<grid, block>>>(Z, b, M, N);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void add_bias_relu_kernel(float *__restrict__ Z, const float *__restrict__ b, int M, int N)
{
    // calculate position in Z
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N)
        return;
    
    float result = Z[row * N + col];
    result += b[col];
    Z[row * N + col] = result > 0 ? result : 0;
}

__host__ void add_bias_relu(float *Z, const float *b, int M, int N)
{
    dim3 block(ADDBIAS_BLOCK, ADDBIAS_BLOCK);
    dim3 grid(CEIL_DIV(N, ADDBIAS_BLOCK), 
            CEIL_DIV(M, ADDBIAS_BLOCK));
    add_bias_relu_kernel<<<grid, block>>>(Z, b, M, N);
    CUDA_CHECK(cudaGetLastError());
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

__global__ void xent_backward_kernel(const float *__restrict__ probs, const int *__restrict__ y, float *__restrict__ dZ, int M, int N, float scale)
{
    // calculate position in dZ
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N)
        return;
    
    dZ[row * N + col] = (probs[row * N + col] - (float)(col == y[row])) * scale;
}

__host__ void xent_backward(const float *probs, const int *y, float *dZ, int M, int N, float scale)
{  
    if (scale <= 0.f)
        scale = 1 / (float)M;
    dim3 block(XENTBWD_BLOCK, XENTBWD_BLOCK);
    dim3 grid(CEIL_DIV(N, XENTBWD_BLOCK), 
            CEIL_DIV(M, XENTBWD_BLOCK));
    xent_backward_kernel<<<grid, block>>>(probs, y, dZ, M, N, scale);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void relu_backward_kernel(float *__restrict__ dZ, const float *__restrict__ A, int M, int N)
{
    // calculate position in dZ
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N)
        return;
    
    dZ[row * N + col] *= (A[row * N + col] > 0.f);
}

__host__ void relu_backward(float *dZ, const float *A, int M, int N)
{
    dim3 block(RELUBWD_BLOCK, RELUBWD_BLOCK);
    dim3 grid(CEIL_DIV(N, RELUBWD_BLOCK), 
            CEIL_DIV(M, RELUBWD_BLOCK));
    relu_backward_kernel<<<grid, block>>>(dZ, A, M, N);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void col_sum_kernel(const float *__restrict__ dZ, float *__restrict__ db, int M, int N)
{
    int column = blockIdx.x;
    int tid = threadIdx.x;

    __shared__ float sm[COLSUM_BLOCK];

    float partial = 0.f;
    for(int i = tid; i < M; i += blockDim.x){
        partial += dZ[i * N + column];
    }
    sm[tid] = partial;
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s >>= 1){
        if (tid < s) sm[tid] += sm[tid + s];
        __syncthreads();
    }
    __syncthreads();
    if(tid == 0)
        db[column] = sm[0]; 
}

__host__ void col_sum(const float *dZ, float *db, int M, int N)
{
    dim3 block(COLSUM_BLOCK);
    dim3 grid(N);
    col_sum_kernel<<<grid, block>>>(dZ, db, M, N);
    CUDA_CHECK(cudaGetLastError());
}
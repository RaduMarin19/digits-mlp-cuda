#pragma once

#include "common.cuh"

#pragma region forward

static constexpr int ADDBIAS_BLOCK = 16;
static constexpr int SOFTMAX_BLOCK = 32;
static constexpr int COLSUM_BLOCK = 256;
static constexpr int XENTBWD_BLOCK = 16;
static constexpr int RELUBWD_BLOCK = 16;

/**
 * Compute Z + b
 * Z: [M x N]
 * B: [N]
 */
__global__ void add_bias_kernel(float* __restrict__ Z,
                        const float* __restrict__ b,
                        int M, int N);

__host__ void add_bias(float* Z,
                    const float* b,
                    int M, int N);
/**
 * Compute relu(Z + b)
 * Z: [M x N]
 * B: [N]
 */
__global__ void add_bias_relu_kernel(float* __restrict__ Z,
                        const float* __restrict__ b,
                        int M, int N);

__host__ void add_bias_relu(float* Z,
                        const float* b,
                        int M, int N);
/**
 * compute softmax for the last layer and return probabilities and loss
 * Z: [M x N]
 * y: [M]
 * probs: [M x N]
 * loss: [M] can be nullptr for inference
 */
__global__ void softmax_xent_kernel(const float* Z, 
                            const int* __restrict__ y, 
                            float* __restrict__ probs, 
                            float* __restrict__ loss, 
                            int M, int N);

__host__ void softmax_xent(const float* Z, 
                        const int* y, 
                        float* probs, 
                        float* loss, 
                        int M, int N);

#pragma endregion

#pragma region backward

/**
 * Gradient of mean cross-entropy w.r.t. the logits.
 *  dZ[i][c] = (probs[i][c] - (c == y[i])) * scale
 * 
 * probs: [M x N] softmax output from softmax_xent, read-only
 * y: [M] true labels
 * dZ: [M x N] output
 * scale: 1/M for batch-mean gradients, 1 for summed
 */
__global__ void xent_backward_kernel(const float* __restrict__ probs,
                            const int* __restrict__ y,
                            float* __restrict__ dZ,
                            int M, int N, float scale);

__host__ void xent_backward(const float* probs, 
                                const int* y,
                                float* dZ,
                                int M, int N, float scale);

/**
 * Gradient of relu w.r.t. the error.
 * dZ: [M x N] gradient 
 * A: [M x N] values of activations, needed for checking which of them were greater than 0
 */
__global__ void relu_backward_kernel(float* __restrict__ dZ, 
                            const float* __restrict__ A,
                            int M, int N);

__host__ void relu_backward(float* dZ, 
                            const float* A,
                            int M, int N);
/**
 * Column-wise sum: db[j] = sum over i for dZ[i][j]
 * that's the bias gradient (broadcast forward -> sum backward)
 * 
 * dZ: [M x N] read-only
 * dB: [N] output
 * 
 * Launch one block per column; the block's threads stride down the rows
 * and combine their partials via a shared-mem tree reduction.
 */
__global__ void col_sum_kernel(const float* __restrict__ dZ,
                        float* __restrict__ db, 
                        int M, int N);

__host__ void col_sum(const float* dZ,
                        float* db, 
                        int M, int N);

#pragma endregion
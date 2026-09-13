#pragma once

#include "common.cuh"

/**
 * Row-major naive GEMM: C = A * B
 * A: [M, K], B: [K, N], C: [M, N]
 */
__global__ void gemm_naive(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ C,
                        int M, int N, int K
                    );

void gemm(const float* A, const float* B, float* C, int M, int N, int K);
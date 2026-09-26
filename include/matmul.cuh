#pragma once

#include "common.cuh"

static constexpr int BLOCK_SIZE = 16;

/**
 * Row-major naive GEMM: C = A * B
 * A: [M, K], B: [K, N], C: [M, N]
 */
template<bool TA, bool TB>
__global__ void gemm_kernel(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ C,
                        int M, int N, int K
                    );

void gemm(const float* A, const float* B, float* C, int M, int N, int K, bool ta = false, bool tb = false);
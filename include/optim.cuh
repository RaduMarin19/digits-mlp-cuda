#include "common.cuh"
/**
 * takes in weight matrix and current derivative and returns updated weights
 * W: [M x N]
 * dW: [M x N]
 * v: [M x N] - velocity, running average of historical gradients
 * lr: float - learning rate, weight of the current derivative in update
 * miu: float - weight of velocity in update
 */
__global__ void sgd_momentum(float* __restrict__ W, 
                            float* __restrict__ v, 
                            const float* __restrict__ dW, 
                            int N, float lr, float miu);
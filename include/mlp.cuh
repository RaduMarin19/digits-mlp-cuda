#pragma once

#include "common.cuh"
#include "matmul.cuh"
#include "layers.cuh"
#include "optim.cuh"

typedef struct{
    int in, out; // layer dimensions

    float *W; // [in, out]
    float *b;  // [out]

    float *dW; // [in, out]
    float *db;  // [out]

    float *vW; // [in, out]
    float *vb; // [out]

    float *A; // [batch, out] 
    float *dZ; // [batch, out]
} Layer;

typedef struct {
    Layer* layers;
    int num_layers;
    int batch_size;
    float* probs;
    float* loss;
} MLP;

void forward(MLP* net, const float* X, const int* y);

void backward(MLP* net);
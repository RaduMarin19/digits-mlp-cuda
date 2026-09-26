#pragma once

#include <random>
#include <algorithm>

#include "common.cuh"
#include "matmul.cuh"
#include "layers.cuh"
#include "optim.cuh"
#include "data.cuh"

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

void backward(MLP* net, const float* X, const int* y);

void mlp_init(MLP* net, const int* dim_layers, int num_layers, int batch_size, int feats, int C);

void mlp_free(MLP* net);

void update(MLP* net, float lr, float miu);

void evaluate(MLP* net, Dataset* ds, float *out_loss, float *out_acc);

void mlp_save(MLP* net, const char *path);

void mlp_load(MLP* net, const char *path);
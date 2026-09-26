#include "mlp.cuh"

void forward(MLP *net, const float *X, const int* y)
{   
    int batch = net->batch_size;
    int n_layers = net->num_layers;
    for (int l = 0; l < n_layers; ++l){
        Layer* L = &net->layers[l];

        const float *A_prev = (l == 0) ? X : net->layers[l-1].A;
        gemm(A_prev, L->W, L->A, batch, L->out, L->in);
        if (l < n_layers - 1)
            add_bias_relu(L->A, L->b, batch, L->out);
        else
            add_bias(L->A, L->b, batch, L->out);
    }
    softmax_xent(net->layers[n_layers-1].A, y, net->probs, net->loss, batch, net->layers[n_layers-1].out);
}

void backward(MLP* net, const float* X, const int* y){
    int batch = net->batch_size;
    int n_layers = net->num_layers;

    for(int l = n_layers - 1; l >= 0; --l){
        Layer* L = &net->layers[l];

        const float* A_prev = (l == 0) ? X : net->layers[l - 1].A;
        
        // produce dZ for this layer
        if (l == n_layers - 1)
            xent_backward(net->probs, y, L->dZ, batch, L->out);
        else
            relu_backward(L->dZ, L->A, batch, L->out);

        // weight and bias gradients
        gemm(A_prev, L->dZ, L->dW, L->in, L->out, batch, true, false);
        col_sum(L->dZ, L->db, batch, L->out);

        // pass gradient to layer below
        if (l > 0)
            gemm(L->dZ, L->W, net->layers[l-1].dZ, batch, L->in, L->out, false, true);
    }
}

void update(MLP* net, float lr, float miu){
    int threads = 256;
    for (int l = 0; l < net->num_layers; ++l){
        Layer *L = &net->layers[l];

        int nW = L->in * L->out;
        int nb = L->out;
        sgd_momentum<<<CEIL_DIV(nW, threads), threads>>>(
            L->W, L->vW, L->dW, nW, lr, miu
        );
        sgd_momentum<<<CEIL_DIV(nb, threads), threads>>>(
            L->b, L->vb, L->db, nb, lr, miu
        );
        CUDA_CHECK(cudaGetLastError());
    }
}

void mlp_init(MLP *net, const int* dim_layers, int num_layers, int batch_size, int feats, int C)
{
    net->batch_size = batch_size;
    net->num_layers = num_layers;
    net->layers = (Layer*)malloc(net->num_layers * sizeof(Layer));
    for(int l = 0; l < net->num_layers; ++l){
        Layer* L = &net->layers[l];
        
        if(l == 0)
            L->in = feats;
        else
            L->in = net->layers[l-1].out;
        if(l == num_layers - 1)
            L->out = C;
        else
            L->out = dim_layers[l];
        CUDA_CHECK(cudaMalloc(&L->b, L-> out * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&L->dW, L->in * L-> out * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&L->db, L-> out * sizeof(float)));

        CUDA_CHECK(cudaMalloc(&L->vW, L->in * L-> out * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&L->vb, L-> out * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&L->A, net->batch_size * L-> out * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&L->dZ, net->batch_size * L-> out * sizeof(float)));

        CUDA_CHECK(cudaMemset(L->b,  0, L->out * sizeof(float)));
        CUDA_CHECK(cudaMemset(L->dW, 0, L->in * L->out * sizeof(float)));
        CUDA_CHECK(cudaMemset(L->db, 0, L->out * sizeof(float)));
        CUDA_CHECK(cudaMemset(L->vW, 0, L->in * L->out * sizeof(float)));
        CUDA_CHECK(cudaMemset(L->vb, 0, L->out * sizeof(float)));

        float* hW = (float*)malloc(L->in * L->out * sizeof(float));
        float std = sqrtf(2.0f / L->in);
        std::mt19937 rng {std::random_device{}()};
        std::uniform_int_distribution<int> dist {0, std};
        std::generate(hW, hW + L->in * L->out * sizeof(float), dist);
        CUDA_CHECK(cudaMemcpy(L->W, hW, L->in * L->out * sizeof(float),
                      cudaMemcpyHostToDevice));
        free(hW);
    }
    CUDA_CHECK(cudaMalloc(&net->probs, net->batch_size * C * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&net->loss, net->batch_size * sizeof(float)));
}

void mlp_free(MLP* net){
    for(int l = 0; l < net->num_layers; ++l){
        Layer* L = &net->layers[l];

        cudaFree(L->W);
        cudaFree(L->b);
        cudaFree(L->dW);
        cudaFree(L->db);
        cudaFree(L->vW);
        cudaFree(L->vb);
        cudaFree(L->A);
        cudaFree(L->dZ);
    }
    cudaFree(net->probs);
    cudaFree(net->loss);
    free(net->layers);
}


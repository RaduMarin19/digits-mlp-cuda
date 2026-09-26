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

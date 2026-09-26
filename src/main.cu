#include "data.cuh"
#include "mlp.cuh"
#include "optim.cuh"

int main(){
    Dataset train, val, test;
    dataset_load(&train, "data/train.bin");
    dataset_load(&test, "data/test.bin");
    dataset_load(&val, "data/val.bin");

    printf("train: %d x %d, %d classes\n", train.n, train.d, train.C);

    int epochs = 5;
    float lr = 0.001;
    float miu = 0.9;
    int n_hidden = 3;
    int hidden[] = {200, 100, 50};
    int batch_size = 64;
    MLP net;
    printf("calling mlp_init...\n"); fflush(stdout);
    mlp_init(&net, hidden, n_hidden, batch_size, train.d, train.C);
    printf("mlp_init done\n"); fflush(stdout);
    int* hIdx = (int*)malloc(train.n * sizeof(int));
    int* dIdx;
    CUDA_CHECK(cudaMalloc(&dIdx, train.n * sizeof(int)));

    for(int epoch = 1; epoch <= epochs; ++epoch){
        printf("epoch: %d\n", epoch);
        for(int s = 0; s < train.n; ++s){
            hIdx[s] = s;
        }

        shuffle(hIdx, train.n);

        CUDA_CHECK(cudaMemcpy(dIdx, hIdx, train.n * sizeof(int), cudaMemcpyHostToDevice));

        float* X_b;
        int* y_b;
        CUDA_CHECK(cudaMalloc(&X_b, batch_size * train.d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&y_b, batch_size * sizeof(int)));

        for (int batch = 0; batch < train.n / batch_size; ++batch) {
            gather_rows(train.X, dIdx + batch*batch_size, X_b, train.d, batch_size);
            gather_labels(train.y, dIdx + batch*batch_size, y_b, batch_size);
            forward(&net, X_b, y_b);
            backward(&net, X_b, y_b);
            update(&net, lr, miu);
        }
        cudaFree(X_b);
        cudaFree(y_b);
        float val_loss, val_acc;
        evaluate(&net, &val, &val_loss, &val_acc);
        printf("validation mean loss: %.4f\n", val_loss);
        printf("validation accuracy: %.4f\n", val_acc);
    }

    mlp_free(&net);
    dataset_free(&train);
    dataset_free(&test);
    dataset_free(&val);
}
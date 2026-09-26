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
    float miu = 0.005;
    MLP net;
    mlp_init(&net); 

    for(int epoch = 1; epoch <= epochs; ++epoch){
        printf("epoch: %d\n", epoch);

    }
}
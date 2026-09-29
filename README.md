# digits-mlp-cuda

A small multilayer perceptron written from scratch in CUDA C++ and trained on a handwritten digits dataset. No PyTorch, no cuBLAS, no cuDNN: the matrix multiplies, activations, loss and gradients are all my own kernels.

I built this to understand what frameworks actually do on the GPU when they train a network. Reading about it was not enough, so I wrote the whole thing myself.

## What it does

- Forward pass: matrix multiply + bias, then ReLU
- Loss: softmax + cross-entropy
- Backward pass: gradients for the weights (`dW`), biases (`db`) and the previous layer's activations (`dA_prev`)
- Parameter update with gradient descent
- Data is downloaded with a Python script in `scripts/` and split 70/15/15 into train / validation / test

## Project layout

```
src/        CUDA kernels and the training loop
include/    headers
tests/      tests for matmul and the layers
scripts/    dataset download + split
data/       dataset
```

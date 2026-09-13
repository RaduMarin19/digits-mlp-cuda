## Scope: build mlp in CUDA and train on digits dataset.

1. Prepare dataset
    1.1. Download data (python script)
    1.2. Split data in 70/15/15

2. MLP should have: 
    2.1. Forward prop: matmul + ReLU
    2.2. Loss: softmax + cross-entropy
    2.3. Backward gradients d_W, d_b, d_Aprev
    2.4. update W += f(dW)

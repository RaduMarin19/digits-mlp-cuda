import os
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

RANDOM_STATE = 32

def save_bin(path, X, y, n_classes):
    with open(path, "wb") as f:
        np.array([X.shape[0], X.shape[1], n_classes], dtype = np.int32).tofile(f)
        X.astype(np.float32).tofile(f)
        y.astype(np.int32).tofile(f)

if __name__ == "__main__":
    os.makedirs("data")
    digits = datasets.load_digits()
    X = digits.data.astype(np.float32)
    y = digits.target.astype(np.int32)
    C = 10

    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state= RANDOM_STATE
    )
    X_va, X_te, y_va, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state= RANDOM_STATE
    )

    for name, (Xs, ys) in {"train": (X_tr, y_tr),
                           "val": (X_va, y_va),
                           "test": (X_te, y_te)}.items():
        save_bin(f"data/{name}.bin", Xs, ys, C)
        print(name, Xs.shape, np.bincount(ys))

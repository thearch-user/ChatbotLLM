import numpy as np

def batch_iterator(X, Y, batch_size, shuffle=True):
    N = X.shape[0]
    indices = np.arange(N)

    if shuffle:
        np.random.shuffle(indices)

    for start in range(0, N, batch_size):
        end = start + batch_size
        batch_idx = indices[start:end]
        yield X[batch_idx], Y[batch_idx]

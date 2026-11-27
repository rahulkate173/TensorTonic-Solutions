import numpy as np

def knn_distance(X_train: list, X_test: list, k: int) -> np.ndarray:
    """
    Returns a NumPy array of neighbor indices with shape (n_test, k).
    If k > n_train, the remaining spots are padded with -1.
    """
    X_train = np.asarray(X_train, dtype=float)
    X_test = np.asarray(X_test, dtype=float)
    
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # Handle 1D feature inputs by reshaping to (n_samples, 1)
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
    if X_test.ndim == 1:
        X_test = X_test.reshape(-1, 1)

    # Compute squared norms with shapes (n_test, 1) and (1, n_train) for broadcasting
    test_sq = np.sum(X_test ** 2, axis=1, keepdims=True)
    train_sq = np.sum(X_train ** 2, axis=1, keepdims=True).T

    # Compute squared Euclidean distance matrix of shape (n_test, n_train)
    # Clip at 0.0 to avoid tiny negative values from floating-point precision
    D_2 = np.maximum(test_sq + train_sq - 2 * (X_test @ X_train.T), 0.0)
    D = np.sqrt(D_2)

    # If k is larger than the number of training samples, we retrieve all available neighbors
    # and then pad the rest with -1.
    effective_k = min(k, n_train)
    neighbor_indices = np.argsort(D, axis=1)[:, :effective_k]

    if k > n_train:
        # Pad each test sample's list of indices with -1
        padding = np.full((n_test, k - n_train), -1, dtype=int)
        neighbor_indices = np.hstack((neighbor_indices, padding))

    return np.asarray(neighbor_indices, dtype=int)
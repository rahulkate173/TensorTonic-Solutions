import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))


def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    samples,features = X.shape
    w = np.zeros(features) # (feature,)
    b = 0.0
    # log reg : (sample ,feature) * (feature,) + 0.0 =(sample,) + 0.0
    for i in range(steps):
        z = (X @ w) + b
        y_pred = _sigmoid(z)

        error = y_pred - y  # [feature,]
        grad_w = (X.T @ error) / samples # (feature,samples) 
        grad_b = float(np.mean(error))

        w -= lr * grad_w
        b -= lr * grad_b

    return w,float(b)
        
        
        
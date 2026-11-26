import numpy as np

def compute_advantage(states: list, rewards: list, V: list, gamma: float) -> np.ndarray:
    """
    Returns the advantages as a NumPy array.
    """
    states = np.asarray(states, dtype=int)
    rewards = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(V, dtype=np.float64)
    
    T = len(rewards)
    returns = np.zeros(T, dtype=np.float64)
    G = 0.0
    
    for t in reversed(range(T)):
        G = rewards[t] + gamma * G
        returns[t] = G
        
    advantages = returns - v[states]
    return np.round(advantages, 4)
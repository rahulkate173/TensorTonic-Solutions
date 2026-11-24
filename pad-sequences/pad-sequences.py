import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    max_len = max(map(len, seqs), default=0) if max_len is None else max_len
    
    # Guarantees (N, L) shape and integer dtype even when seqs is empty
    out = np.full((len(seqs), max_len), pad_value, dtype=np.int64)
    
    for i, seq in enumerate(seqs):
        trunc = seq[:max_len]
        out[i, :len(trunc)] = trunc
        
    return out

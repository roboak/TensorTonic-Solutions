import numpy as np

def pad_sequences(seqs: list, pad_value: int = 0, max_len: int | None = None) -> np.ndarray:
    """
    Returns: np.ndarray of shape (N, L) where:
      N = len(seqs)
      L = max_len if provided else max(len(seq) for seq in seqs) or 0
    """
    # Your code here
    if not seqs:
        return np.array([], dtype=int).reshape(0, 0)

    N = len(seqs)
    L = max_len if max_len else max(len(seq) for seq in seqs)
    padded_array = np.full((N, L), pad_value)
    for i, seq in enumerate(seqs):
        padded_array[i,:min(len(seq), L)] = seq[:min(len(seq), L)]

    return padded_array

import numpy as np

def dropout(
    x: list,
    p: float = 0.5,
    rng: np.random.Generator = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (output, dropout_pattern) as NumPy arrays matching the shape of x.
    """
    # Write code here
    x = np.asarray(x)
    mask = rng.random(x.shape) if rng else np.random.random(x.shape)
    mask = np.where(mask>p, 1/(1-p), 0)
    output = x*mask

    return output, mask
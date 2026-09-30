import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    # Write code here
    # sigmoid = 1/(1 + np.exp(-1*np.array(x)))
    x = -1*np.array(x)
    return (1/(1+np.exp(np.array(x))))

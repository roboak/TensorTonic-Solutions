import numpy as np
def value_iteration_step(values: list, transitions: list, rewards: list, gamma: float) -> list[float]:
    """
    Returns one updated floating-point value for every state.
    """
    # Write code here
    # [S,A,S)]
    values= np.asarray(values)
    transitions = np.asarray(transitions)
    rewards = np.asarray(rewards)
    values_new = np.max(rewards + gamma*transitions@values, axis=1)

    return list(values_new)
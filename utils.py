import math
import numpy as np


def round_integer_part(point, integer_indices, mode='floor', lb=None, ub=None):
    """
    Rounds specified integer components of a point according to the chosen mode ('floor', 'ceil', 'round'),
    respecting variable lower and upper bounds if provided.
    """
    rounded = np.array(point, dtype=float).copy()
    for idx in integer_indices:
        val = rounded[idx]
        if mode == 'floor':
            value = math.floor(val)
        elif mode == 'ceil':
            value = math.ceil(val)
        elif mode == 'round':
            value = round(val)
        else:
            value = math.floor(val)

        if lb is not None and lb[idx] is not None:
            value = max(value, lb[idx])
        if ub is not None and ub[idx] is not None:
            value = min(value, ub[idx])
        rounded[idx] = value
    return rounded

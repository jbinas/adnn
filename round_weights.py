
import numpy as np

def round_weights(X, int_bits, frac_bits, ifactor=1., prog_bits=False):
    fractional_shift = 2.**(frac_bits)
    max_limit = (2.**(int_bits + frac_bits))-1.
    value = np.where(X < 0, X / ifactor, X * 1.)
    value = value * fractional_shift
    value = np.round(value)
    value = np.clip(value, -max_limit, max_limit)
    value = value / fractional_shift
    if not prog_bits:
        value = np.where(value < 0, value * ifactor, value)
    return value

from numpy import linalg as LA
import numpy as np
import cmath
import Func_Unit1 as funcs


n_bits = 4
def diffusion_matrix():
    """Return the diffusion matrix.

    Returns:
        array[float]: The matrix representation of the diffusion operator.
    """
    ##################
    # YOUR CODE HERE #
    ##################
    matrix = 1/(2**(n_bits-1)) * np.ones(2 ** n_bits) - np.eye(2 ** n_bits)
    
    return matrix

print(np.sqrt(5))
n_bits = 5
query_register = list(range(n_bits))
aux = [n_bits]
all_wires = query_register + aux


print(query_register)
print(len(query_register))
print(np.zeros(len(query_register)))
print(np.pi/4)
print(3/2)
combos = [[int(s) for s in np.binary_repr(j, n_bits)] for j in range(2**4)]
print(combos)
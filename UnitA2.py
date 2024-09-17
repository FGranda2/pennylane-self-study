import pennylane as qml
import numpy as np

def oracle_matrix(combo):
    """Return the oracle matrix for a secret combination.
    
    Args:
        combo (list[int]): A list of bits representing a secret combination.
         
    Returns: 
        array[float]: The matrix representation of the oracle.
    """
    index = np.ravel_multi_index(combo, [2]*len(combo)) # Index of solution
    print('Index:')
    print(index)
    my_array = np.identity(2**len(combo)) # Create the identity matrix

    ##################
    # YOUR CODE HERE #
    ##################
    my_array[index,index]=-my_array[index,index]

    # MODIFY DIAGONAL ENTRY CORRESPONDING TO SOLUTION INDEX

    return my_array

combo=[1,0]
print(oracle_matrix(combo))
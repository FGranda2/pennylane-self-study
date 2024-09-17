from numpy import linalg as LA
import numpy as np
import cmath
import Func_Unit1 as funcs

X = np.array([[0,1],[1,0]])

eigenvalues, eigenvectors = LA.eig(X)

# print("The eigenvalues are: ", eigenvalues)
# print("The eigenvectors are: ", eigenvectors)

# print("Result: ", np.array([1,0])*eigenvectors[0])

def is_hermitian(matrix):
    """Check whether a matrix is hermitian.
    
    Args:
        matrix: (array(array[complex]))
    Returns:
        bool: True if the matrix is Hermitian, False otherwise
    """
 
    ##################
    # YOUR CODE HERE #
    ################## 
    # eigenvalues, eigenvectors = LA.eig(matrix)
    # matrix = np.array([[eigenvalues[0],0],[0,eigenvalues[1]]])
    # print(eigenvalues)
    eigenvalues, eigenvectors = LA.eigh(matrix)
    if not np.array_equal(np.matrix(matrix).getH(),matrix):
        return False
    # matrix = np.array([[eigenvalues[0],0],[0,eigenvalues[1]]])
    print(eigenvalues)
    if np.trace(matrix) != 1:
        print("trace ", np.trace(matrix))
        return False
    
    eigenvalues, eigenvectors = LA.eig(matrix)
    for val in eigenvalues:
        if val < 0:
            return False
    
    return True# Return the boolean value

matrix_1 = np.array([[1,1j],[-1j,1]])
matrix_2 = np.array([[1,2],[3,4]])

print("Is matrix [[1,1j],[-1j,1]] Hermitian?")
print(is_hermitian(matrix_1))
print("Is matrix [[1,2],[3,4]] Hermitian?")
print(is_hermitian(matrix_2))



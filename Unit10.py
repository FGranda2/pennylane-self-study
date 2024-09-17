from numpy import linalg as LA
import numpy as np
import cmath
import Func_Unit1 as funcs

state = np.array([4/5, -3*np.exp(1j*np.pi/3)/5])
B = np.array([[1,-2j],[2j,2]])

eigenvalues, eigenvectors = LA.eig(B)

print("EigenVals: ", np.real(eigenvalues))
print("EigenStates: ", eigenvectors)
from numpy import linalg as LA
import numpy as np
import cmath
import Func_Unit1 as funcs

X=np.array([[0,1],[1,0]])

eigenvalues, eigenvectors = LA.eig(X)

print("The eigenvalues are: ", eigenvalues)
print("The eigenvectors are: ", eigenvectors)
print("sqrt result ",1/np.sqrt(2))

state_plus = np.array([1/np.sqrt(2),1/np.sqrt(2)])
state_minus = np.array([1/np.sqrt(2),-1/np.sqrt(2)])

norm_plus = funcs.inner_product(state_plus,state_plus)
norm_minus = funcs.inner_product(state_minus,state_minus)

print("Norm +: ", norm_plus, "Norm -: ", norm_minus)

orthogonal = funcs.inner_product(state_plus,state_minus)
print("Orthogonal: ", orthogonal)
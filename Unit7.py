from numpy import linalg as LA
import numpy as np
import cmath
import Func_Unit1 as funcs

np.set_printoptions(suppress=True)
print("1/root(2) is: ", 1/np.sqrt(2))
theta = np.pi/2
rx = np.array([[np.cos(theta/2),np.sin(theta/2)*1j ],[-np.sin(theta/2)*1j,np.cos(theta/2)]])

phi = np.pi/2
rz1 = np.array([[1,0],[0, np.exp(phi*1j)]])

omega = -3*np.pi/2
rz2 = np.array([[1,0],[0, np.exp(omega*1j)]])
print("RX:")
print(rx)

print("RZRX:")
print(np.matmul(rz1,rx))

print("---RZRX:")
print(np.matmul(rx,rz1))

print("RZRXRZ:")
print(np.matmul(np.matmul(rz2,rx), rz1))
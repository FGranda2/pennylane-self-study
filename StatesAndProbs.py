from numpy import linalg as LA
import numpy as np
import cmath
import Func_Unit1 as funcs
import json
import pennylane as qml


dev = qml.device("default.qubit", wires=2)

@qml.qnode(dev)
def circuit(angles):
    """
    The quantum circuit that you will simulate.

    Args:
        angles (list(float)): The gate angles in the circuit.

    Returns:
        (numpy.tensor): The probability vector of the underlying quantum state
        that this circuit produces.
    """

    qml.RY(angles[0],0)
    qml.RY(angles[1],1)

    return qml.probs(wires=[0, 1])
import pennylane as qml
import numpy as np

dev = qml.device("default.qubit", wires=2)


@qml.qnode(dev)
def circuit_1():
    
    qml.X(0)
    qml.CNOT([0,1])

    return qml.state()

print("AFTER: ")
print(circuit_1())
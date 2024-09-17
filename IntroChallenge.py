import json
import pennylane as qml
import pennylane.numpy as np
def W(params):
    
    """
    Subcircuit that implements the trainable block W
    
    Args:
        params (np.array): A matrix containing the parameters for the trainable block W. The length of
        params is equal to the depth of the circuit. The length of each row in params is the number 
        of qubits used. See the challenge statement for a detailed explanation
    Returns:
        Since this function is a subcircuit, you must not return anything.
    
    """
    layers, wires = params.shape
    for layer in range(layers):
        # Apply the rotation with the respective angles
        for wire in range(wires):
            angle_rad = params[layer][wire]
            qml.RY(angle_rad,wire)

        # Apply the entanglement
        for wire in range(wires):
            if (wire + 1 == wires):
                qml.CNOT([wire,0])
            else:
                qml.CNOT([wire,wire+1])


    

    # Put your code here
    

def S(g, x, num_wires):
    
    """
    Subcircuit that implements the encoding block S
    
    Args:
        g (pennylane.Operator): A PennyLane operator representing the generator for the encoding
        gates. It must be Hermitian in order to generate a unitary. Call it as g(wires) to specify
        the wires on which it acts.
        x (complex): The scalar coefficient of the operator g.
        num_wires (int): The number of wires over which the encoding gate is broadcast.
        
        
    Returns:
        Since this function is a subcircuit, you must not return anything.
    
    """
    for wire in range(num_wires):
        op = qml.exp(g(wire), 0.5j * x)
        qml.QubitUnitary(op.matrix(),wire, unitary_check=True)   



# Create a device

dev = qml.device("default.qubit", wires=range(4), shots=None) 

@qml.qnode(dev, expansion_strategy = "device")
def quantum_model(param_set, g, x):
    
    """
    This QNode implements the quantum model with alternating trainable and encoding blocks
    
    Args:
        param_set (np.array): A numpy array whose elements are the parameter matrices for each of the trainable
        blocks. Therefore, the length of this list is equal to the number of trainable blocks, which is greater
        than, or equal to 2.
        g (pennylane.Operator): A PennyLane operator representing the generator for the encoding
        gates. It must be Hermitian in order to generate a unitary.
        x: The scalar coefficient of the operator g.
    Returns:
        (np.tensor(float)): A tensor of dimensions (2,) representing the measurement probabilities in the computational 
        basis on the first wire.
    """
    

    # Put your code here
    trainable, layers, wires = param_set.shape
    for i,param in enumerate(param_set):
        if (i == trainable-1):
            W(param)

            
        else:
            W(param)
            S(g,x,wires)


    
    return qml.probs(0)
# These functions are used to test your solution

def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    params = np.array(ins[0])
    g = getattr(qml, ins[1])
    x = ins[2]
    outs = quantum_model(params, g, x).tolist()
    return str(outs)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    
    dev_test = qml.device("default.qubit", wires = [0,1,2])    
    
    @qml.qnode(dev_test)
    def w_node(params):
        
        W(params)
        
        return qml.probs(wires = [0,1]) 
    
    @qml.qnode(dev_test)
    def s_node(g, x, num_wires):
        
        S(g, x, num_wires)
        
        return qml.probs(wires = [0,1]) 
        
    params_test = np.array([[np.pi, np.pi/4, np.pi],[np.pi, np.pi/4, np.pi/3]])
    w_test = w_node(params_test)
    
    s_test = s_node(qml.PauliX, np.pi/7, 3)
    
    assert np.allclose(w_test, [0.10983496, 0.21338835, 0.03661165, 0.64016504], atol = 1e-3), "Something isn't quite right with the trainable block."

    assert np.allclose(s_test, [0.65892978, 0.15281512, 0.15281512, 0.03543998], atol = 1e-3), "Something isn't quite right with the encoding block."

    assert np.allclose(solution_output,expected_output, atol = 1e-3), "Not the correct probabilities for the quantum model."

# These are the public test cases
test_cases = [
    ('[[[[1.0472, 0.7854, 3.1416, 0.3927],[1.0472, 0.7854, 3.1416, 0.5236]],[[1.0472, 0.7854, 1.5708, 0.3927],[0.7854, 0.7854, 1.5708, 0.7854]]],"PauliX", 0.7854]', '[0.46653, 0.53347]'),
    ('[[[[0.62832, 0.3927, 1.0472, 0.7854],[0.7854, 0.31416, 0.62832, 0.5236]],[[0.31416, 0.7854, 0.7854, 0.3927],[0.31416, 0.3927, 0.31416, 0.3927]]],"PauliY", 0.5236]', '[0.68594, 0.31406]')
]
# This will run the public test cases locally
for i, (input_, expected_output) in enumerate(test_cases):
    print(f"Running test case {i} with input '{input_}'...")

    try:
        output = run(input_)

    except Exception as exc:
        print(f"Runtime Error. {exc}")

    else:
        if message := check(output, expected_output):
            print(f"Wrong Answer. Have: '{output}'. Want: '{expected_output}'.")

        else:
            print("Correct!")
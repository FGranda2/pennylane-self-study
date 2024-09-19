import json
import pennylane as qml
import pennylane.numpy as np

import json
import pennylane as qml
import pennylane.numpy as np
def W(alpha, beta):
    """ This function returns the matrix W in terms of
    the coefficients alpha and beta

    Args:
        - alpha (float): The prefactor alpha of U in the linear combination, as in the
        challenge statement.
        - beta (float): The prefactor beta of V in the linear combination, as in the
        challenge statement.
    Returns 
        -(numpy.ndarray): A 2x2 matrix representing the operator W,
        as defined in the challenge statement
    """

    # Put your code here #
    # Return the real matrix of the unitary W, in terms of the coefficients.
    U = 1/np.sqrt(alpha+beta) * np.array([[np.sqrt(alpha),-np.sqrt(beta)],[np.sqrt(beta),np.sqrt(alpha)]])
    return U

dev = qml.device('default.qubit', wires = 2)

@qml.qnode(dev)
def linear_combination(U, V,  alpha, beta):
    """This circuit implements the circuit that probabilistically calculates the linear combination 
    of the unitaries.

    Args:
        - U (list(list(float))): A 2x2 matrix representing the single-qubit unitary operator U.
        - V (list(list(float))): A 2x2 matrix representing the single-qubit unitary operator U.
        - alpha (float): The prefactor alpha of U in the linear combination, as above.
        - beta (float): The prefactor beta of V in the linear combination, as above.

    Returns:
        -(numpy.tensor): Probabilities of measuring the computational
        basis states on the auxiliary wire. 
    """

    # Put your code here #
    # Return the probabilities on the first wire
    # Get the W Matrix
    W_mat = W(alpha,beta)

    # Apply it to bit 0
    qml.QubitUnitary(W_mat, wires=0)

    # Apply an X gate to bit 0
    qml.X(0)

    # Apply the controlled U-Gate
    qml.ControlledQubitUnitary(U,0,1)

    qml.X(0)

    # Apply the controlled V-Gate
    qml.ControlledQubitUnitary(V,0,1)

    # Apply it to bit 0
    qml.QubitUnitary(np.transpose(np.conj(W_mat)), wires=0)
    result = qml.probs(0)
    # print(result)
    return result
    
    

# These functions are responsible for testing the solution.

def run(test_case_input: str) -> str:
    dev = qml.device('default.qubit', wires = 2)
    ins = json.loads(test_case_input)
    output = linear_combination(*ins)[0].numpy()
    print("THE PROBS ARE: ", output)

    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    assert np.allclose(
        solution_output, expected_output, atol=1e-3
    ), "Your circuit doesn't look quite right "

# These are the public test cases
test_cases = [
    ('[[[ 0.70710678,  0.70710678], [ 0.70710678, -0.70710678]],[[1, 0], [0, -1]], 1, 3]', '0.8901650422902458')
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
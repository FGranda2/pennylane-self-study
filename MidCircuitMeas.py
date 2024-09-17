import json
import pennylane as qml
import pennylane.numpy as np
n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def circuit(angles):
    """A quantum circuit made from the quantum function U.

    Args:
        angles (list(float)): A list of angles containing theta_0, theta_1, theta_2, and theta_3 in that order. 
    Returns:
        (numpy.tensor): The probability of the fourth qubit.
    """

    # Put your code here #
    # The Hadamard to all
    for i in range(n_qubits):
        qml.Hadamard(i)

    # The RX to wire 0
    qml.RX(angles[0],0)

    # The CNOTs
    qml.CNOT([0,3])
    qml.CNOT([2,1])

    # Do the measurements 
    meas_0 = qml.measure(0)
    meas_2 = qml.measure(2)

    # Apply the U3 if condition is met
    qml.cond(meas_0 + meas_2 >= 1, qml.U3)(angles[1],angles[2],angles[3], 3)

    # Do the PauliZ
    qml.Z(3)
    return qml.probs(wires=3)

# These functions are responsible for testing the solution.
def run(test_case_input: str) -> str:
    angles = json.loads(test_case_input)
    output = circuit(angles).tolist()
    return str(output)

def check(solution_output: str, expected_output: str) -> None:
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)

    assert np.allclose(solution_output, expected_output, rtol=1e-4)

# These are the public test cases
test_cases = [
    ('[1.0, 1.5, 2.0, 2.5]', '[0.79967628, 0.20032372]')
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
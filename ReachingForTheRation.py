import json
import pennylane as qml
import pennylane.numpy as np

from pennylane import qaoa
# Write any helper functions you need here

def qaoa_layer(h_cost,h_mixer, gamma, alpha):
    qml.ApproxTimeEvolution(h_cost,gamma,1)
    qml.ApproxTimeEvolution(h_mixer,alpha,1)

def cost_hamiltonian(edges):
    """
    This function build the QAOA cost Hamiltonian for a graph, encoded in its edges

    Args:
    - Edges (list(list(int))): A list of ordered pairs of integers, representing 
    the edges of the graph for which we need to solve the minimum vertex cover problem.

    Returns:
    - pennylane.Operator: The cost Hamiltonian associated with the graph.
    """
    

    # Put your code here #
     
    # First, get the unique vertices set
    unique_vertices = set()
    for edge in edges:
        # Add each value in the element to the set (which keeps only unique values)
        unique_vertices.update(edge)

    unique_vertices = list(unique_vertices)

    # Compute the second term of the Hamiltonian
    ones_vector_coeff = [1] * len(unique_vertices)
    obs = [qml.PauliZ(i) for i in unique_vertices]
    H_second = qml.dot(ones_vector_coeff,obs)

    # Compute the first term of the Hamiltonian
    ones_vector_coeff = [1] * len(edges)
    obs = [qml.PauliZ(i) @ qml.PauliZ(j) + qml.PauliZ(i) + qml.PauliZ(j) for i,j in edges]
    H_first = qml.dot(ones_vector_coeff,obs)

    return 3/4 * H_first - H_second


def mixer_hamiltonian(edges):
    """
    This function build the QAOA mixer Hamiltonian for a graph, encoded in its edges

    Args:
    - edges (list(list(int))): A list of ordered pairs of integers, representing 
    the edges of the graph for which we need to solve the minimum vertex cover problem.

    Returns:
    - pennylane.Operator: The mixer Hamiltonian associated with the graph.
    """

    # Put your code here #
    # First, get the unique vertices set
    unique_vertices = set()
    for edge in edges:
        # Add each value in the element to the set (which keeps only unique values)
        unique_vertices.update(edge)

    unique_vertices = list(unique_vertices)

    # Compute the second term of the Hamiltonian
    ones_vector_coeff = [1] * len(unique_vertices)
    obs = [qml.PauliX(i) for i in unique_vertices]
    H_mixer = qml.dot(ones_vector_coeff,obs)

    return H_mixer

def qaoa_circuit(params, edges):
    """
    This quantum function (i.e. a list of gates describing a circuit) implements the QAOA algorithm
    You should only include the list of gates, and not return anything

    Args:
    - params (np.array): A list encoding the QAOA parameters. You are free to choose any array shape you find 
    convenient.
    - edges (list(list(int))): A list of ordered pairs of integers, representing 
    the edges of the graph for which we need to solve the minimum vertex cover problem.

    Returns:
    - This function does not return anything. Do not write any return statements.
    
    """

    # Put your code here
    depth = 4

    h_cost = cost_hamiltonian(edges)
    h_mixer = mixer_hamiltonian(edges)

    unique_vertices = set()
    for edge in edges:
        # Add each value in the element to the set (which keeps only unique values)
        unique_vertices.update(edge)

    unique_vertices = list(unique_vertices)

    # Compute the second term of the Hamiltonian
    wires = range(len(unique_vertices))

    for w in wires:
        qml.Hadamard(wires=w)
    # print(params[0],params[1])
    qml.layer(qaoa_layer, depth, [h_cost,h_cost,h_cost,h_cost], [h_mixer,h_mixer, h_mixer,h_mixer], params[0], params[1])
    
    

# This function runs the QAOA circuit and returns the expectation value of the cost Hamiltonian

dev = qml.device("default.qubit")

@qml.qnode(dev)
def qaoa_expval(params, edges):
    qaoa_circuit(params, edges)
    return qml.expval(cost_hamiltonian(edges))

def optimize(edges):
    """
    This function returns the parameters that minimize the expectation value of
    the cost Hamiltonian after applying QAOA

    Args:
    - edges (list(list(int))): A list of ordered pairs of integers, representing 
    the edges of the graph for which we need to solve the minimum vertex cover problem.

    
    """

    # Your cost function should be the expectation value of the cost Hamiltonian
    # You may use the qaoa_expval QNode defined above
    
    # Write your optimization routine here
    opt = qml.GradientDescentOptimizer()
    steps = 300

    params = np.array([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]], requires_grad=True)
    for i in range(steps):
        params = opt.step(qaoa_expval, params, edges)
        params = params[0]

    print("Optimal Parameters")
    print(params)
    return params
    
    

# These are auxiliary functions that will help us grade your solution. Feel free to check them out!

@qml.qnode(dev)
def qaoa_probs(params, edges):
  qaoa_circuit(params, edges)
  return qml.probs()

def approximation_ratio(params, edges):

    true_min = np.min(qml.eigvals(cost_hamiltonian(edges)))

    approx_ratio = qaoa_expval(params, edges)/true_min

    return approx_ratio

# These functions are responsible for testing the solution.

def run(test_case_input: str) -> str:
    ins = json.loads(test_case_input)
    params = optimize(ins)
    output= approximation_ratio(params,ins).numpy()

    ground_energy = np.min(qml.eigvals(cost_hamiltonian(ins)))

    index = np.argmax(qaoa_probs(params, ins))
    vector = np.zeros(len(qml.matrix(cost_hamiltonian(ins))))
    vector[index] = 1

    calculate_energy = np.real_if_close(np.dot(np.dot(qml.matrix(cost_hamiltonian(ins)), vector), vector))
    verify = np.isclose(calculate_energy, ground_energy)

    if verify:
      return str(output)
    
    return "QAOA failed to find right answer"

def check(solution_output: str, expected_output: str) -> None:

    assert not solution_output == "QAOA failed to find right answer", "QAOA failed to find the ground eigenstate."
        
    solution_output = json.loads(solution_output)
    expected_output = json.loads(expected_output)
    print("Solution output: ", solution_output)
    print("Expected output: ", expected_output)
    assert solution_output >= expected_output-0.01, "Minimum approximation ratio not reached"

# These are the public test cases
test_cases = [
    ('[[0, 1], [1, 2], [0, 2], [2, 3]]', '0.55'),
    ('[[0, 1], [1, 2], [2, 3], [3, 0]]', '0.92'),
    ('[[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 4]]', '0.55')
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
import numpy as np
import cmath

# Here are the vector representations of |0> and |1>, for convenience
ket_0 = np.array([1, 0])
ket_1 = np.array([0, 1])


def normalize_state(alpha, beta):
    """Compute a normalized quantum state given arbitrary amplitudes.

    Args:
        alpha (complex): The amplitude associated with the |0> state.
        beta (complex): The amplitude associated with the |1> state.

    Returns:
        np.array[complex]: A vector (numpy array) with 2 elements that represents
        a normalized quantum state.
    """

    ##################
    # YOUR CODE HERE #
    ##################
    k_abs = 1 / (np.sqrt(np.power(abs(alpha),2,dtype=complex) + np.power(abs(beta),2,dtype=complex)))
    
    # CREATE A VECTOR [a', b'] BASED ON alpha AND beta SUCH THAT |a'|^2 + |b'|^2 = 1

    # RETURN A VECTOR
    return k_abs*np.array([alpha,beta])

def inner_product(state_1, state_2):
    """Compute the inner product between two states.

    Args:
        state_1 (np.array[complex]): A normalized quantum state vector
        state_2 (np.array[complex]): A second normalized quantum state vector

    Returns:
        complex: The value of the inner product <state_1 | state_2>.
    """

    ##################
    # YOUR CODE HERE #
    ##################
    result = np.conj(state_1[0]) * state_2[0] + np.conj(state_1[1]) * state_2[1]

    # COMPUTE AND RETURN THE INNER PRODUCT

    return result

def measure_state(state, num_meas):
    """Simulate a quantum measurement process.

    Args:
        state (np.array[complex]): A normalized qubit state vector.
        num_meas (int): The number of measurements to take

    Returns:
        np.array[int]: A set of num_meas samples, 0 or 1, chosen according to the probability
        distribution defined by the input state.
    """

    # COMPUTE THE MEASUREMENT OUTCOME PROBABILITIES
    prob_0 = np.real(state[0]*np.conj(state[0]))
    prob_1 = 1-prob_0

    # GET SAMPLES USING NUMPY RANDOM CHOICE
    result = np.empty(num_meas, dtype=int)
    a = np.array([0,1])
    for i in range(num_meas):
        sample = np.random.choice(a=a, p = np.array([prob_0, prob_1]))
        result[i] = sample
    # RETURN A LIST OF SAMPLE MEASUREMENT OUTCOMES
    return result

def apply_u(state):
    """Apply a quantum operation.

    Args:
        state (np.array[complex]): A normalized quantum state vector.

    Returns:
        np.array[complex]: The output state after applying U.
    """

    result = np.matmul(state,U)

    # APPLY U TO THE INPUT STATE AND RETURN THE NEW STATE
    return result

alpha_test = -1.83697020e-6-1.j
beta_test = -1.46957616e-5-8.j

result = normalize_state(alpha_test,beta_test)

print(result[0],result[1])

# EX I.1.3
state = np.array([1/2,np.sqrt(3)*1j/2])

result = inner_product(state,state)
print("The product is: ", result)
print("The probability of 1 is: ", state[1]*np.conj(state[1]))

result = measure_state(state,10)
print("Measured State: ", result)

U = np.array([[0,1j],[-1j,0]])

result = apply_u(state)
print("The multi result is: ", np.real(result[0]))
print("The multi result is: ", result[1])

print("prob_0 is: ", np.real(result[0]*np.conj(result[0])))
print("prob_1 is: ", np.real(result[1]*np.conj(result[1])))
print("prob_ sum is: ", np.real(result[0]*np.conj(result[0]))+np.real(result[1]*np.conj(result[1])))

testMat = np.array([['a','b'],['c','d']])
testMat_T = np.array([['a_star','c_star'],['b_star','d_star']])
print("Matrix: ", testMat)
print("Matrix: ", testMat_T)
print("Matrix Transpose: ", np.matmul(testMat,testMat_T))
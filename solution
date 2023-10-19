import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def gru(inputs, h_prev, Wz, Uz, Wr, Ur, W, U):
    hidden_states = []
    for inp in inputs:
        z = sigmoid(np.dot(Wz, inp) + np.dot(Uz, h_prev))
        r = sigmoid(np.dot(Wr, inp) + np.dot(Ur, h_prev))
        h_tilda = tanh(np.dot(W, inp) + np.dot(U, r * h_prev))
        h_next = z * h_prev + (1 - z) * h_tilda
        hidden_states.append(h_next)
        h_prev = h_next
    return hidden_states


#exmple
# Initializing inputs
inputs = [np.random.rand(5) for _ in range(10)]
h_prev = np.zeros(5)

# Initializing weights
Wz, Uz, Wr, Ur, W, U = [np.random.rand(5, 5) for _ in range(6)]

# Call gru function
hidden_states = gru(inputs, h_prev, Wz, Uz, Wr, Ur, W, U)

# Print the output
for i, h in enumerate(hidden_states):
    print(f"Hidden state at time {i}: {h}")

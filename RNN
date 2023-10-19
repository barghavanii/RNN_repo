import numpy as np

def rnn_backpropagation(W, U, V, x, y, k, g, g_prime, e_prime):
    # Initialize the weights
    dW = np.zeros_like(W)
    dU = np.zeros_like(U)
    dV = np.zeros_like(V)

    # Initialize hiddenstate and delta_h
    h = np.zeros((W.shape[0], 1))
    delta_h = np.zeros_like(h)

    # backpropagat through time
    for t in reversed(range(k)):
        # output of the RNN
        a = np.dot(W, h) + np.dot(U, x[:, t].reshape(-1, 1))
        h = g(a)

        # error of output layer
        error = y[:, t].reshape(-1, 1) - np.dot(V, h)

        # delta  hidden state
        delta_h = np.dot(V.T, error) + np.dot(W.T, delta_h) * g_prime(a)

        # Update the weights
        dW += np.dot(delta_h, h.T)
        dU += np.dot(delta_h, x[:, t].reshape(-1, 1).T)
        dV += np.dot(error, h.T)

    return dW, dU, dV
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1 - np.tanh(x)**2

# derivetive for error
def mse_derivative(y_true, y_pred):
    return 2 * (y_pred - y_true)
k = 5
input_size = 3
hidden_size = 4
output_size = 2

x = np.random.rand(input_size, k)
y = np.random.rand(output_size, k)

W = np.random.rand(hidden_size, hidden_size)
U = np.random.rand(hidden_size, input_size)
V = np.random.rand(output_size, hidden_size)

grad_W, grad_U, grad_V = rnn_backpropagation(W, U, V, x, y, k, tanh, tanh_prime, mse_derivative)

print("Grad W:")
print(grad_W)
print("\nGrad U:")
print(grad_U)
print("\nGrad V:")
print(grad_V)

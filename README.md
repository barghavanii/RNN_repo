# Problem 1. Gated Recurrent Unit

-Implement a Gated Recurrent Unit (GRU) in Python using numpy.
-Given a sequence of inputs and corresponding hidden states, the GRU should compute the output hidden states at each time step using the equations pro- vided above.
-Implement a function that takes in the input sequence inputs, initial hidden state h_prev, and the weight matrices Wz, Uz, Wr, Ur, W, and U as inputs and returns the sequence of hidden states.
-The GRU should use sigmoid as the activation function for the update and reset gates, and tanh as the activation function for the candidate hidden state.
# Problem 2. Long Short-Term Memory


• Implement a LSTM cell in Python using the equations provided in the lecture. • Create a LSTM network with one LSTM cell and pass a sequence of random input data through it. • Print the output hidden state h and memory cell c at each time step.

 # Problem 3. Recurrent Neural Network
Implement a function rnn_backpropagation(W, U, V, x, y, k, g, g_prime, e_prime) that performs the RNN backward propagation through time algorithm described in the text. The function should take in the following inputs: • W: The weight matrix for the hidden units moving forward in time • U: The weight matrix for the input units • V: The weight matrix for the output units • x: The input sequence • y: The true output sequence • k: The length of the input and output sequences • g: The non-linear activation function • g_prime: The derivative of the non-linear activation function • e_prime: The derivative of the error function The function should return a tuple of the gradients for W, U, and V.

# Problem 4. Recurrent Neural Network
Implement a bidirectional RNN in Python. The network should take in a sequence of inputs and output a prediction for each timestep. The network should have the following architecture: • An input layer that takes in a sequence of vectors of length n. • A forward LSTM layer with h hidden units. • A backward LSTM layer with h hidden units. • A concatenation layer that concatenates the outputs from the forward and backward LSTM layers. • A fully connected layer that outputs a prediction for each timestep. The network should be trained on a dataset of sequences and corresponding labels. The loss function used for training should be the mean squared error between the network’s predictions and the true labels. For this I try to use imdb dataset to implement this part


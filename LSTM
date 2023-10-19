import numpy as np

# Define the LSTM cell
class LSTMCell:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # LSTM cell parameters
        self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
        self.Wo = np.random.randn(hidden_size, input_size + hidden_size)
        self.bf = np.zeros((hidden_size, 1))
        self.bi = np.zeros((hidden_size, 1))
        self.bc = np.zeros((hidden_size, 1))
        self.bo = np.zeros((hidden_size, 1))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x, prev_h, prev_c):
        # Concatenate the input and previous hidden state
        concat_input = np.vstack((prev_h, x))
        ft = self.sigmoid(np.dot(self.Wf, concat_input) + self.bf)#forget gate
        it = self.sigmoid(np.dot(self.Wi, concat_input) + self.bi)#input gate
        c_hat = self.tanh(np.dot(self.Wc, concat_input) + self.bc)#candidate cell satet
        c = ft * prev_c + it * c_hat # Cell state
        ot = self.sigmoid(np.dot(self.Wo, concat_input) + self.bo) #output gate
        h = ot * self.tanh(c) # hidden state
        return h, c
#example
# Create an LSTM cell
lstm_cell = LSTMCell(input_size=3, hidden_size=4)

# Create a sequence of random input data
input_sequence = [np.random.randn(3, 1) for _ in range(5)]

# Initialize hidden state and memory cell
h = np.zeros((4, 1))
c = np.zeros((4, 1))

# Process the input sequence through the LSTM cell
for t, x in enumerate(input_sequence):
    h, c = lstm_cell.forward(x, h, c)
    print(f"Time step {t + 1} - Hidden State (h):\n{h}")
    print(f"Time step {t + 1} - Cell State (c):\n{c}")
    print("\n")

from keras.models import Sequential
from keras.layers import Bidirectional, LSTM, Dense, Embedding, Concatenate, Input, Lambda
from keras.losses import MeanSquaredError
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences
from keras import backend as K

# Load and preprocess the IMDb dataset
features = 2000
maxlen = 50
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=features)

X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

# Convert target data to float32
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Define the bidirectional RNN model with concatenation
embedding_size = 128
hidden_units = 64

input_layer = Input(shape=(maxlen,))
embedding_layer = Embedding(input_dim=features, output_dim=embedding_size, input_length=maxlen)(input_layer)
lstm_forward = LSTM(hidden_units, return_sequences=True)(embedding_layer)
lstm_backward = LSTM(hidden_units, return_sequences=True, go_backwards=True)(embedding_layer)

# Concatenate the outputs from forward and backward LSTMs
concatenated = Concatenate()([lstm_forward, lstm_backward])

output_layer = Dense(1, activation='linear')(concatenated)  # Linear activation for regression

model = Model(inputs=input_layer, outputs=output_layer)

# Define a custom loss function for mean squared error
def custom_mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred), axis=-1)

# Compile the model with the custom mean squared error (MSE) loss function
model.compile(optimizer='adam', loss=custom_mean_squared_error, metrics=['mean_squared_error'])

# Train the model on the IMDb dataset
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=64)

# You can evaluate the model and make predictions as needed.

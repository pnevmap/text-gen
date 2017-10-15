import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# DONE: fill out the function below that transforms the input series
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []

    i = 0
    while i + window_size < len(series):
        X.append(series[i:(i + window_size)])
        i += 1

    y = series[window_size:]

    # reshape each
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:window_size])
    y = np.asarray(y)
    y.shape = (len(y), 1)

    return X, y

# DONE: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))
    return model
    pass


### DONE: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    import re
    return re.sub(r'[^a-z!,.:;?]', ' ', text)


### DONE: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    i = 0

    while i + window_size < len(text):
        inputs.append(text[i:i + window_size])
        outputs.append(text[i + window_size])
        i += step_size

    return inputs, outputs

# DONE build the required RNN model:
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))
    return model

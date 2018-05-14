import numpy as np
import string
import re

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = []
    y = []
   
    for i in range(0, len(series) - window_size):
        X.append(series[i:i+window_size]) # exemple: X.append(series[0:7])
        y.append(series[i+window_size]) # exemple: y.append(series[7])

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    # Getting started with the Keras Sequential model
    model = Sequential()
    # layer 1 uses an LSTM module with 5 hidden units (note here the input_shape = (window_size,1))
    model.add(LSTM(5, input_shape=(window_size, 1)))
    # layer 2 uses a fully connected module with one unit
    model.add(Dense(1))
    # return RNN model
    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']
        
    # replace '\n' and other escapes
    text = text.replace('\a', ' ')
    text = text.replace('\b', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\f', ' ')
    text = text.replace('\t', ' ')
    text = text.replace('\v', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')    
     
    # only ascii lowercase
    text = ascii(text).lower()    

    # only punctuation given included and ascii alphabet
    text = re.sub('[^a-z \! \, \. \: \; \?]', '', text)
    
    # shorten any extra dead space created above
    text = text.replace('     ',' ')
    text = text.replace('    ',' ')
    text = text.replace('   ',' ')
    text = text.replace('  ',' ')
    text = text.replace('  ',' ')    
    
    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []
    
    for i in range(0, len(text) - window_size, step_size):
        inputs.append(text[i:i+window_size]) # exemple: inputs.append(text[0:100])
        outputs.append(text[i+window_size]) # exemple: outputs.append(text[100])       

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    # Getting started with the Keras Sequential model
    model = Sequential()
    # layer 1 should be an LSTM module with 200 hidden units --> note this should have input_shape = (window_size,len(chars))
    # where len(chars) = number of unique characters in your cleaned text
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    # layer 2 should be a linear module, fully connected, with len(chars) hidden units --> where len(chars) = number of unique characters
    # in your cleaned text
    model.add(Dense(num_chars))
    # layer 3 should be a softmax activation (since we are solving a multiclass classification)
    model.add(Activation('softmax'))
    # return RNN model
    return model

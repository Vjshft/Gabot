import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from tensorflow.python.lib.io import file_io
import urllib


text = urllib.urlopen("http://investors.mx/textos/ojos-de-perro-azul.txt").read().decode('UTF-8')
text = text.lower()

characters = sorted(list(set(text)))

n_to_char = {n: char for n, char in enumerate(characters)}
char_to_n = {char: n for n, char in enumerate(characters)}

X = []
Y = []
length = len(text)
seq_length = 100

for i in range(0, length - seq_length, 1):
    sequence = text[i:i + seq_length]
    label = text[i + seq_length]
    X.append([char_to_n[char] for char in sequence])
    Y.append(char_to_n[label])

X_modified = np.reshape(X, (len(X), seq_length, 1))
X_modified = X_modified / float(len(characters))
Y_modified = np_utils.to_categorical(Y)

model = Sequential()
model.add(LSTM(300, input_shape=(X_modified.shape[1], X_modified.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(300, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(300))
model.add(Dropout(0.2))
model.add(Dense(Y_modified.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')
model.fit(X_modified, Y_modified, epochs=50, batch_size=200)

model.save("gabo_model.h5")
model.save_weights("gabo_weights.h5")

with file_io.FileIO('gabo_model.h5', mode='r') as input_f:
        with file_io.FileIO('gs://model-bot/gabo_model.h5', mode='w+') as output_f:
            output_f.write(input_f.read())

with file_io.FileIO('gabo_weights.h5', mode='r') as input_f:
        with file_io.FileIO('gs://model-bot/gabo_weights.h5', mode='w+') as output_f:
            output_f.write(input_f.read())

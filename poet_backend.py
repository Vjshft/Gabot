import flask
import numpy as np
import tensorflow as tf
from keras.models import load_model
from flask import jsonify
from flask import request
from flask_cors import CORS

# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
CORS(app)

def init():
    global model, graph

    # load the pre-trained Keras model
    model = load_model('models/gabo_model.h5')
    model.load_weights('models/gabo_weights.h5')
    graph = tf.get_default_graph()

@app.route("/predict", methods=["POST"])
def predict():
    print("predict")
    entrada = request.form['frase']
    texto = predecir(entrada)
    res = {"texto":texto}
    res = jsonify(res)
    return res

def predecir(entrada):
    text = (open("texts/ojos-de-perro-azul.txt").read())
    text = text.lower()
    entrada = entrada.lower()
    characters = sorted(list(set(text)))
    n_to_char = {n:char for n, char in enumerate(characters)}
    char_to_n = {char:n for n, char in enumerate(characters)}

    X = []
    length = len(text)
    seq_length = 100
    sequence = entrada[0:seq_length]
    X.append([char_to_n[char] for char in sequence])

    string_mapped = X[0]
    full_string = [n_to_char[value] for value in string_mapped]
    # generating characters
    for i in range(500):
        x = np.reshape(string_mapped,(1,len(string_mapped), 1))
        x = x / float(len(characters))
        with graph.as_default():
            pred_index = np.argmax(model.predict(x, verbose=0))
        seq = [n_to_char[value] for value in string_mapped]
        full_string.append(n_to_char[pred_index])
        print(string_mapped)
        string_mapped.append(pred_index)
        string_mapped = string_mapped[1:len(string_mapped)]
    txt=""
    for char in full_string:
        txt = txt+char
    txt = txt+"..."
    print(txt)
    return txt

# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
"please wait until server has fully started"))
    init()
    app.run(threaded=True, host='0.0.0.0')

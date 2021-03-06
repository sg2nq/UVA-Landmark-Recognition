from flask import Flask, request, jsonify
import tensorflow as tf
import flask

from tensorflow import keras
from tensorflow.keras.models import Model

import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from PIL import Image
import requests
from io import BytesIO

from tensorflow.keras.preprocessing import image
import os

app = Flask(__name__)
cwd_path = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(os.path.join(cwd_path, 'LandmarkRecognizer.h5'))
@app.route('/predict', methods=['GET'])


def predict():
    if request.method == 'GET':
        data = flask.request.query_string[4:]
        if (data == None):
            return flask.jsonify({"Error": "Missing params"})
        # if parameters are found, echo the msg parameter 
        if (data != None):
            print(data)

    response = requests.get(data)
    img = Image.open(BytesIO(response.content))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    class_names = ['AcademicalVillage', 'AldermanLibrary', 'AlumniHall', 'AquaticFitnessCenter', 
    'BararoHall', 'BrooksHall', 'ClarkHall', 'MadisonHall', 'MinorHall', 'NewCabellHall', 
    'NewcombHall', 'OldCabellHall', 'OlssonHall', 'RiceHall', 'Rotunda', 'ScottStadium', 
    'ThorntonHall', 'UniversityChapel']
    results = model.predict(x)
    return jsonify({"prediction": class_names[results.argmax(axis=-1)[0]]})
# start the flask app, allow remote connections
app.run(host='0.0.0.0', port = 8080)

# Note: For the purposes of this demo, MobileNet V2 was used as the base model as its lightweight (18MB) and meets github's filesize limit.

from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
from torchvision import transforms

import flask
import numpy as np
import os
import requests
import torch
import torch.nn.functional as F

app = Flask(__name__)
cwd_path = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cpu")
model = torch.load(os.path.join(cwd_path, 'uvalandmarkmodel.pth'), map_location=torch.device('cpu'))
model = model.to(device)
class_names = ['AcademicalVillage', 'AldermanLibrary', 'AlumniHall', 'AquaticFitnessCenter', 'BravoHall', 'BrooksHall', 'ClarkHall', 'JeffersonHall', 'MadisonHall', 'MinorHall', 'NewCabellHall', 'NewcombHall', 'OldCabellHall', 'OlssonHall', 'PavillionVII', 'RiceHall', 'Rotunda', 'ScottStadium', 'ThorntonHall', 'UVAHistoricalLandmarkRecognitionAppBackgroundcopypng', 'UniversityChapel']

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
    preprocessFn = transforms.Compose(
        [transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize(mean = [0.485, 0.456, 0.406], 
            std = [0.229, 0.224, 0.225])])
    inputs =  preprocessFn(img).unsqueeze(0)
    inputs = inputs.to(device)
    outputs = model(inputs)
    probs, indices = (-F.softmax(outputs, dim = 1).data).sort()
    probs = (-probs).numpy()[0]; indices = indices.numpy()[0]
    preds = {class_names[idx]:float(prob) for (prob, idx) in zip(probs, indices)}
    return jsonify({"prediction": preds})
# start the flask app, allow remote connections
app.run(host='0.0.0.0', port = 8080)

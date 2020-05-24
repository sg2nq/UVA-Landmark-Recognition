# UVaLandmarkRecognitionTransferLearning

# Hosting A Model Trained using Transfer Learning with Tensorflow 2.0 and Keras

## Part 1 - Transfer Learning
### Import tensorflow 2.0
```try:
  # Use the %tensorflow_version magic if in colab.
  %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf
```

### Download, Unzip and import the dataset
Ideally, the dataset should have files in ./train  and ./test to make importing the files easier. Alternately, different methods can be used to import the images rather than flow_from_directory

To download and unzip simply make a wget request to the dataset and unzip the downloaded file using command line(!):
```
!wget -O "dataset.zip" "https://www.dropbox.com/s/qdptwne9j43z70d/dataset_split.zip"
!unzip "/content/dataset.zip"
```
Using Image data generator and flow_from_directory from tf.keras.preprocessing, import the dataset (and repeat for the validation set):
```
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
# note that the preprocess_input should be from the pretrained model being used 
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies
train_generator=train_datagen.flow_from_directory('/content/train',
                                                 target_size=(224,224),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)
```
### Load the pretrained model of choice
Choose a model from the following link: https://www.tensorflow.org/api_docs/python/tf/keras/applications
In the sample code, we chose to use the Inception V3 model. 
First, we need to import the different objects we are going to need:
```
import pandas as pd
import numpy as np
import os
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```

Load the model, add a Global Average Pooling layer to the output, followed by a Densely conneced network, with softmax activation. The softmax layer will give the prediction output. The first step of the process only trains the added layer, which is the dense layer that was added at the end. Thus, we initially keep the base layer frozen by setting all layers in the base untrainable.

```
base_model=InceptionV3(weights='imagenet',include_top=False)
for layer in base_model.layers:
    layer.trainable = False

x=base_model.output
x=GlobalAveragePooling2D()(x)
preds=Dense(18,activation='softmax')(x)

model=Model(inputs=base_model.input,outputs=preds)
```

### Compile, Train, and Test the model
Compile the model with the settings of your choice, for the optimizer, loss and metrics. Then, use model.fit to train it.

We chose to use the Stochastic Gradient Descent optimzier with a learning rate of 0.1, momentum of 0.9, and decay of 0.01 for the first step. We chose 10 epochs for the training as after 8 there is a mild decrease in accuracy, and this verifies that we have achieved the best results possible with the given hyperparameters.

The step size is the number of batches in the dataset.
```
from tensorflow.keras import optimizers

sgd = optimizers.SGD(lr=0.1, momentum=0.9, decay = 0.01)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size
val_step_size=val_generator.n//val_generator.batch_size

model.fit(train_generator, steps_per_epoch=step_size_train,
                    validation_data=val_generator, 
                    validation_steps = val_step_size,
                    epochs=10)
```
The second step retrains the entire model, but at a slower rate so as to not void the pre-trained model. Thus the learning rate was reduced to 0.001 and the decay to 0.0001.
```
for layer in model.layers:
  layer.trainable = True

sgd = optimizers.SGD(lr=0.001, momentum=0.9, decay = 0.0001, nesterov = True)
model.compile(optimizer=sgd,loss='categorical_crossentropy',metrics=['accuracy'])
step_size_train=train_generator.n//train_generator.batch_size
val_step_size=val_generator.n//val_generator.batch_size
model.fit(train_generator, steps_per_epoch=step_size_train,
                    validation_data=val_generator, 
                    validation_steps = val_step_size,
                    epochs=10)
```
### Export the model
Save the model using model.save. Make sure that the model is an object of class tensorflow.keras.models.Model.
```
model.save('LandmarkRecognizer.h5')  
```

## Part 2 - Hosting the Model Using Flask, Google Cloud Platform, Docker, and Kubernetes

### Server
After saving the model, we created a Flask server in Python.

To do so, first we need to install all dependencies (Insturctions for linux below):
Install python3.7, pip, and python3 setuptools for your machine. Ensure that pip is at its latest version. Install Flask, Tensorflow, Pillow and requests.
Flask is used for making the server, tensorflow is required to import the model and make the prediction, Pillow enables us to load the image, and requests is used to download the image from the internet.
```
apt-get update
apt-get install -y python3.7
apt-get install -y python3-pip
apt-get install -y python3-setuptools
python3.7 -m pip install --upgrade pip
python3.7 -m pip install flask
python3.7 -m pip install tensorflow
python3.7 -m pip install Pillow
python3.7 -m pip install requests
```
Once all the dependencies are downloaded without errors, we can make the Flask API. First import all the required classes from the liraries that were installed.
```
from flask import Flask, request, jsonify
import tensorflow as tf
import flask

from tensorflow import keras
from tensorflow.keras.models import Model

import numpy as np
from tensorflow.keras.applications.resnet import preprocess_input

from PIL import Image
import requests
from io import BytesIO

from tensorflow.keras.preprocessing import image
import os

```

Initialize the flask app, load the model and set the app route to https://<link>/predict

```
app = Flask(__name__)
file_path = os.path.dirname(os.path.abspath(__file__))

model = tf.keras.models.load_model(os.path.join(file_path, 'LandmarkRecognizer.h5'))
@app.route('/predict', methods=['GET'])

```
The first part of the predict function gets the query string from the http request, and stores it in the variable data.
```
def predict():
    if request.method == 'GET':
        data = flask.request.query_string[4:]
        if (data == None):
            return flask.jsonify({"Error": "Missing params"})
        # if parameters are found, echo the msg parameter 
        if (data != None):
            print(data)
```

The second part of the predict function downloads the image from the link provided and pre-processes it in the same manner as training time. Then, it calls the model to make the prediction, and returns the result in JSON format.
```
    response = requests.get(data)
    img = Image.open(BytesIO(response.content))
    # img = image.load_img(test_file_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    class_names = ['AcademicalVillage', 'AldermanLibrary', 'AlumniHall', 'AquaticFitnessCenter', 
    'BravoHall', 'BrooksHall', 'ClarkHall', 'MadisonHall', 'MinorHall', 'NewCabellHall', 
    'NewcombHall', 'OldCabellHall', 'OlssonHall', 'RiceHall', 'Rotunda', 'ScottStadium', 
    'ThorntonHall', 'UniversityChapel']
    results = model.predict(x)
    return jsonify({"prediction": class_names[results.argmax(axis=-1)[0]]})
# start the flask app, allow remote connections
app.run(host='0.0.0.0', port = 8080)
```
You can test the API by locally running the flask server. Simply call `python3 api.py`. This will create a local server, which can be accessed at:
```http://0.0.0.0:8080/predict?msg=```
Insert the URL to the image after "msg="

### Containerize and Upload to GCP Kubernetes
To create the docker image, we first make the docker file. Simply create file called `Dockerfile` with the following contents:
```
FROM ubuntu:latest
MAINTAINER Sid  
RUN apt-get update \  
  && apt-get install -y python3.7 \
  && apt-get install -y python3-pip \
  && apt-get install -y python3-setuptools \
  && cd /usr/local/bin \  
  && ln -s /usr/bin/python3.7 python \  
  && python3.7 -m pip install --upgrade pip \
  && python3.7 -m pip install flask \
  && python3.7 -m pip install tensorflow \
  && python3.7 -m pip install Pillow \
  && python3.7 -m pip install requests


COPY api.py api.py
COPY LandmarkRecognizer.h5 LandmarkRecognizer.h5
ENTRYPOINT ["python3.7","api.py"]
```
This will take care of importing all the needed libraries into the docker image, as well as the api file and the model.

Finally, create the docker image and upload the image to GCP. Follow the instructions here:
[https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app](https://cloud.google.com/kubernetes-engine/docs/tutorials/hello-app)
Brief overview of the steps, but full instructions should be read to avoid errors:
Run the following command inside the folder containing the docker file
```docker build -t gcr.io/uva-landmark-images/ml-model:v<Current Version> .```

Configure docker
```gcloud auth configure-docker```

Test the docker locally
```
docker push gcr.io/uva-landmark-images/ml-model:v<Current Version>
docker run --rm -p 8080:8080 gcr.io/uva-landmark-images/ml-model:v<Current Version>
```
Follow the same URL as for flask to test it locally.

Update the kubernetes image to the one created:
```
kubectl set image deployment/ml-web-model ml-model=gcr.io/uva-landmark-images/ml-model:v<Current Version>
```

Set of details for Kubernetes Cluster:
Project ID: uva-landmark-images
Cluster name: ml-cluster
Workload name: ml-web-model
Container name: ml-model
Endpoint: 35.231.33.254:80

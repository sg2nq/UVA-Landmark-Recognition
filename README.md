# UVaLandmarkRecognitionTransferLearning

## Transfer Learning using Tensorflow 2.0 and Keras

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
Using Image data generator and flow_from_directory from tf.keras.preprocessing, import the dataset (and repeat for the test set):
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
In the sample code, we chose to use the Xception model. 
First, we need to import the different objects we are going to need:
```
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
```

Load the model, add a Global Average Pooling layer to the output, followed by 2 Densely conneced networks, one with activation relu and the other with softmax. The softmax layer will give the actual output.
```
base_model=Xception(weights='imagenet',include_top=False)
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
preds=Dense(18,activation='softmax')(x) #final layer with softmax activation
model=Model(inputs=base_model.input,outputs=preds)
```

### Compile, Train, and Test the model
Compile the model with the settings of your choice, for the optimizer, loss and metrics. Then, use model.fit to train it.
```
model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_generator, steps_per_epoch=train_generator.n//train_generator.batch_size, epochs=10)
```
To test the model, call model.evaluate and it will return the result based on the metrics chosen during compilation:
```
model.evaluate(test_generator)
```
### Export the model
Save the model using model.save. Make sure that the model is an object of class tensorflow.keras.models.Model.
```
model.save('model_name.h5')  
```
### Convert the model into a tensorflow JS model
First, you need to install the command line tool for tensorflowjs:
```
!pip install tensorflowjs
```
Now, use tensorflowjs_converter to convert the keras model to a tensorflow js ready folder:
```
!tensorflowjs_converter --input_format keras /content/model_name.h5 /content/output_folder_name
```

### Final Steps
Once you have the folder created, download and host the folder to a service where you can use a file path. An instance of such a cloud service is Github. The file path system, which does not function with Google Drive, Box, and DropBox, is essential, as tensorflowjs only takes in the URL to the json file created, and the binary files are pulled in subsequent request using the file location. 

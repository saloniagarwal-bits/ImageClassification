import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from flask import Flask, redirect, request, render_template
from werkzeug.utils import secure_filename

import sys
import os
import glob
import re


app = Flask(__name__)
model_path = 'vgg19.h5'

#load model
model = load_model(model_path)
model.make_predict_function()

# we will pre process the image and then predict using model
def model_predict(img_path, model):
    # For Imagenet the image size should always be 224,224
    img = image.load_img(img_path, target_size=(224, 224))

    #preprocess the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    x=  preprocess_input(x)
    preds = model.predict(x)
    return preds


@app.route('/', methods = ['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method=='POST':
        #get the file from the post call. the input name in html file is "inp"
        f = request.files['file']

        # save the image file to upload folder
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # model prediction
        # this will give class index
        pred = model_predict(file_path, model)
        #for imagenet we have to decode the predicted value/class, this will map the output value from imagenet to name of class
        # this will map class index to class label
        pred_class = decode_predictions(pred, top=1)
        result = str(pred_class[0][0][1]) # convert to string
        return result
    return None



if __name__ == '__main__':
    app.run(debug=True)
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 13:06:38 2019

@author: Devesh Kaushik
"""

from flask import Flask, render_template,request, jsonify

import keras
import keras.models
import numpy as np
from PIL import Image
import PIL
import sys
import os

#tell our app where our saved model is
sys.path.append(os.path.abspath("./model"))

from load import * 

app = Flask(__name__)

global model

model = init()

@app.route('/')
def index():
	#initModel()
	#render out pre-built HTML file right on the index page
	return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        #file_name = file.filename
        list_vectors = []
        
        img  = Image.open(file)
        img = img.resize((64, 64), PIL.Image.ANTIALIAS)
        img_array = np.array(img)
        list_vectors.append(img_array)
        
        X = np.stack((list_vectors))
        X = X/255
        
        #prediction:-
        pred_arr = model.predict(X)
        maxElement = np.amax(pred_arr)
        
        class_prediction = np.where(pred_arr == maxElement)
        
        return jsonify({'pred_class': class_prediction, 'probability':maxElement*100})


if __name__ == "__main__":
	app.run()

# Fixed my error with the graph element https://kobkrit.com/tensor-something-is-not-an-element-of-this-graph-error-in-keras-on-flask-web-server-4173a8fe15e1
import keras as kr
# import flask
# import render template which will display the HTML in the browser
# import request will contain all the data that is sent from client to server
from flask import Flask, render_template, request 
# provides functions to encode binary data to Base64
import base64
# this will take care of compiling the model and any saved training configuration's
from keras.models import load_model
# this will help to plot the data
import numpy as np
# imports Python Imaging Library
# Image will be used to open the Image
# ImageFilter will be used to filter the image
from PIL import Image, ImageOps, ImageFilter
# this import is library of Python bindings designed to solve computer vision problems 
import cv2
# is a library for machine learning algorithms
import tensorflow as tf

# create's an instance of the Flask class for our web app.
app = Flask(__name__)

# setting the height for re-sizeing the image for the MNIST datset
HEIGHT = 28
# setting the width for re-sizeing the image for the MNIST datset
WIDTH = 28
SIZE = HEIGHT, WIDTH

# this is the route for the home page 
@app.route('/')
def homePage():
    # this will return the html page fro the template folder
    return render_template('DigitRecognizer.html')

@app.route('/predict', methods=['POST'])
def predict():

    # this will get the image from the request
    encodeThatImage = request.values[('imgBase64')]
    # this will decode the data URL 
    decodeThatImage = base64.b64decode(encodeThatImage[22:])

    # this will save the image 
    with open('DrawnNum.png', 'wb') as f:
        f.write(decodeThatImage)

    # this will open the image 
    predictedImage = Image.open("DrawnNum.png")

    # smooth's the image before resizing
    SmoothIMG = predictedImage.filter(ImageFilter.SMOOTH_MORE)
    # Returns a sized and cropped version of the image
    NewImage = ImageOps.fit(SmoothIMG, SIZE)
    # this will save the rezied image that was created
    NewImage.save("DrawnNumResized.png")
    # Cv2 will load the new resized image
    cv2NewImage = cv2.imread("DrawnNumResized.png")
    # this will convert the the cv2 image to grayscale 
    grayScaleNewImage = cv2.cvtColor(cv2NewImage, cv2.COLOR_BGR2GRAY)
    # converting to float 32 and dividing it by 255 
    grayScaleNewImageArray = np.array(grayScaleNewImage, dtype=np.float32).reshape(1, 784)
    grayScaleNewImageArray /= 255

    # Loading model into memory
    model = load_model('model/NewModel.h5')
    # Passes the array into the model
    SetNewPrediction = model.predict(grayScaleNewImageArray)
    GetNewPrediction = np.array(SetNewPrediction[0])
    # Returning this respone as a string
    NewpredictedNumber = str(np.argmax(GetNewPrediction))
    # Prints what the model is predicting
    print("The nerual network predicted" + NewpredictedNumber)
    # returns the predicted number
    return NewpredictedNumber
    # Runs the application on the local development server
    app.run()
    

   
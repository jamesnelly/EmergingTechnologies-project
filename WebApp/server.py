import keras as kr
from flask import Flask, render_template, request 
from io import BytesIO
import base64
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2
import tensorflow as tf



app = Flask(__name__)

model = load_model('model/NewModel.h5')

HEIGHT = 28
WIDTH = 28
SIZE = HEIGHT, WIDTH

@app.route('/')
def homePage():
    return render_template('DigitRecognizer.html')


@app.route('/predict', methods=['POST'])
def predict():

    encodeThatImage = request.values[('imgBase64')]

    decodeThatImage = base64.b64decode(encodeThatImage[22:])

    with open('DrawnNum.png', 'wb') as f:
        f.write(decodeThatImage)

    predictedImage = Image.open("DrawnNum.png")


    #smooth's the image before resizing
    SmoothIMG = predictedImage.filter(ImageFilter.SMOOTH_MORE)
    # Returns a sized and cropped version of the image
    NewImage = ImageOps.fit(SmoothIMG, SIZE)

    NewImage.save("DrawnNumResized.png")

    cv2NewImage = cv2.imread("DrawnNumResized.png")
    grayScaleNewImage = cv2.cvtColor(cv2NewImage, cv2.COLOR_BGR2GRAY)
    grayScaleNewImageArray = np.array(grayScaleNewImage, dtype=np.float32).reshape(1, 784)
    grayScaleNewImageArray /= 255

    
    SetNewPrediction = model.predict(grayScaleNewImageArray)
    GetNewPrediction = np.array(SetNewPrediction[0])

    NewpredictedNumber = str(np.argmax(GetNewPrediction))
    print(NewpredictedNumber)

    return NewpredictedNumber

    app.run()
    

   
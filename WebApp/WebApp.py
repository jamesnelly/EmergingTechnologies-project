
from flask import Flask, render_template, request
import re
from io import BytesIO
import base64
from keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
import cv2
#import tensorflow as tf



app = Flask(__name__)

model = load_model('model/model.h5')

imageHeight = 28
imageWidth = 28
size = imageHeight, imageWidth

@app.route('/')
def homePage():
    return render_template('WebApp/static/DigitRecognizer.html')

    

if __name__ == '__main__':
    app.run(debug=False, threaded=False)
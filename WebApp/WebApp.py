from flask import Flask, jsonify, request, send_from_static

from scipy.misc import imread, imresize

import tensorflow as tf
from keras.models import load_model

import keras as kr
import numpy as np
import io
import base64

app = Flask(__name__)

global model, graph

def Get_Model():
    mod = load_model('../model/model.h5')
    print('Model loaded')
    return mod

    
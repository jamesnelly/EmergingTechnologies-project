#adapted from  https://www.youtube.com/watch?v=n5a0WBIQitI

#loading the dataset
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

import tensorflow as tf
from keras import models
from keras import layers
import keras as kr
import numpy as np
import matplotlib.pyplot as plt

#creating the sequential model
mod = kr.models.Sequential()

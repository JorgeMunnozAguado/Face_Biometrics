# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 11:46:20 2021

@author: aytha
"""

import keras
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import tensorflow as tf
import keras_vggface
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Lambda, Activation, ActivityRegularization
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers, models, layers, regularizers
from keras.preprocessing import image
from keras_vggface import utils
from keras_vggface.vggface import VGGFace



#Import the ResNet-50 model trained with VGG2 database
my_model = 'resnet50'
resnet = VGGFace(model = my_model)
#resnet.summary()  

#Select the lat leayer as feature embedding  
last_layer = resnet.get_layer('avg_pool').output
feature_layer = Flatten(name='flatten')(last_layer)
model_vgg=Model(resnet.input, feature_layer)


#How to read an input image
from scipy import misc
image_size = 224
img = misc.imread(file_path, mode='RGB')
img = misc.imresize(img, (image_size, image_size), interp='bilinear')
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
embedding = model_vgg.predict(img)


# TASK 1: Read the DiveFace database and obtain the embeddings of 50 face images (1 image per subject) from 
# the 6 demographic groups (50*6=300 embeddings in total).

# Link to the database: https://dauam-my.sharepoint.com/:u:/g/personal/aythami_morales_uam_es/ERd0YZG26FlGl1hr9nQtd54BNmW2XMwuzS-LXh0DoMp2ig?e=f8jD7w


#TASK 2: Using t-SNE, represent the embeddings and its demographic group. Can you differenciate the different demographic groups?

from sklearn.manifold import TSNE


# TASK 3: Using the ResNet-50 embedding (freeze the model), train your own attribute classifiers (ethnicity and gender). 
# Recommendation: use a simple dense layer with a softmax output. Divide DiveFace into train and test.



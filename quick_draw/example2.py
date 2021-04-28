# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 21:01:47 2020

@author: aytha
"""
from __future__ import print_function
from keras.datasets import mnist
from sklearn.manifold import TSNE

from sklearn.svm import SVC

import random
from random import randrange


import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, BatchNormalization
from keras import backend as K
from keras.utils import np_utils
from keras.layers import PReLU
from keras.initializers import Constant
from keras.callbacks import EarlyStopping
import numpy as np
from itertools import permutations
import seaborn as sns
import matplotlib.patheffects as PathEffects
import os
import pickle
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE





def scatter(x, labels, subtitle=None,M=14):
    # We choose a color palette with seaborn.
    palette = np.array(sns.color_palette("hls", M))

    # We create a scatter plot.
    f = plt.figure(figsize=(7, 7))
    ax = plt.subplot(aspect='equal')
    sc = ax.scatter(x[:,0], x[:,1], lw=0, s=40,
                    c=palette[labels.astype(np.int)])
    plt.xlim(-25, 25)
    plt.ylim(-25, 25)
    ax.axis('off')
    ax.axis('tight')

   
    if subtitle != None:
        plt.suptitle(subtitle)
        
    plt.savefig(subtitle)


def data_generator_clusters(X, batch_size,NUM_SAMPLES_USER):
        
    # Create empty arrays to contain batch of features and labels#
    batch_features_positive = np.zeros((batch_size, 28, 28,1))
    batch_features_negative = np.zeros((batch_size, 28, 28,1))
    
    batch_features_positive2 = np.zeros((batch_size, 28, 28,1))
    batch_features_negative2 = np.zeros((batch_size, 28, 28,1))       
        
    batch_labels = np.zeros((batch_size,1))
    M=X.shape[0]
    
    while True:
        for i in range(batch_size):            
            #Elegimos el usuario genuino
            genuine_user = random.choice(range(M))            
            
            index_sample = random.sample(range(0,NUM_SAMPLES_USER), NUM_SAMPLES_USER)#np.random.randint(NUM_SAMPLES_USER)                
             # La muestra positiva siempre del usuario genuino
            genuine_samples = X[genuine_user]
            positive_sample = genuine_samples[index_sample[0]][:,:,:]
            #Zero padding
            batch_features_positive[i] = positive_sample          
            
            # La muestra negativa puede ser de un usuario impostor o del genuinao
            batch_labels[i] = 1#np.random.randint(2)
            if batch_labels[i] == 0: #Muestra del usuario genuino 
                negative_sample = genuine_samples[index_sample[0]][:,:,:]                
            else: #Muestra del usuario impostor                        
                impostor_user = random.choice(range(M))
                
                while genuine_user == impostor_user:
                    impostor_user = random.choice(range(M))                                        
                
                impostor_samples = X[impostor_user]
                negative_sample = impostor_samples[index_sample[0]][:,:,:]
            #Zero padding
            batch_features_negative[i] = negative_sample
            
                
            ######################   2    ####################################    
            positive_sample = genuine_samples[index_sample[1]][:,:,:]
            #Zero padding
            batch_features_positive2[i] = positive_sample  
            
            if batch_labels[i] == 0: #Muestra del usuario genuino               
                
                negative_sample = genuine_samples[index_sample[1]][:,:,:]                
            else: #Muestra del usuario impostor                
                while genuine_user == impostor_user:
                    impostor_user = random.choice(range(M))                                        
                
                impostor_samples = X[impostor_user]
                negative_sample = impostor_samples[index_sample[1]][:,:,:]
            #Zero padding
            batch_features_negative2[i] = negative_sample 
            
           
                   
                
        yield ({'Up_input': batch_features_positive,
                'Down_input': batch_features_negative,
                'Up_input2': batch_features_positive2,
                'Down_input2': batch_features_negative2,                              
                'label': batch_labels}, None)
    
    
    
def contrastive_loss_triplet(inputs, dist='euclidean', margin= 'maxplus',  alpha = 0.5):  #1.5
    Up_output, Down_ouput, Up_output2, Down_ouput2, Y_true = inputs

    Up=Up_output #Anchor
    Down=Down_ouput #Negative
    Up2=Up_output2 #Positive

    distance_p1_up1 = K.square(Up - Up2)   #Anchor-Positive distance
    distance_n1_up1 = K.square(Up - Down)  #Anchor- Negative distance
   
    distance_p1_up1 = K.sqrt(K.sum(distance_p1_up1, axis=-1, keepdims=True))        
    distance_n1_up1 = K.sqrt(K.sum(distance_n1_up1, axis=-1, keepdims=True))  
    
    loss = K.maximum(0.0, (distance_p1_up1) - (distance_n1_up1) + alpha)  
    
    return K.sum(loss)



N=50 #Número de muestras por clase a usar
M=10 #Número de clases
N_test=200 #Numero de muestras de test
img_rows, img_cols = 28, 28

#Load images
X1=np.load('full_numpy_bitmap_aircraft carrier.npy')
X2=np.load('full_numpy_bitmap_airplane.npy')
X3=np.load('full_numpy_bitmap_alarm clock.npy')
X4=np.load('full_numpy_bitmap_ambulance.npy')
X5=np.load('full_numpy_bitmap_angel.npy')
X6=np.load('full_numpy_bitmap_animal migration.npy')
X7=np.load('full_numpy_bitmap_ant.npy')
X8=np.load('full_numpy_bitmap_anvil.npy')
X9=np.load('full_numpy_bitmap_apple.npy')
X10=np.load('full_numpy_bitmap_arm.npy')
#X11=np.load('full_numpy_bitmap_asparagus.npy')
#X12=np.load('full_numpy_bitmap_axe.npy')
#X13=np.load('full_numpy_bitmap_backpack.npy')
#X14=np.load('full_numpy_bitmap_banana.npy')
#X15=np.load('full_numpy_bitmap_bandage.npy')
#X16=np.load('full_numpy_bitmap_The Great Wall of China.npy')
#X17=np.load('full_numpy_bitmap_The Eiffel Tower.npy')
#X18=np.load('full_numpy_bitmap_book.npy')
#X19=np.load('full_numpy_bitmap_barn.npy')
#X20=np.load('full_numpy_bitmap_bird.npy')

#Reshape images into 2D space
X1=X1.reshape(X1.shape[0], img_rows, img_cols)
X2=X2.reshape(X2.shape[0], img_rows, img_cols)
X3=X3.reshape(X3.shape[0], img_rows, img_cols)
X4=X4.reshape(X4.shape[0], img_rows, img_cols)
X5=X5.reshape(X5.shape[0], img_rows, img_cols)
X6=X6.reshape(X6.shape[0], img_rows, img_cols)
X7=X7.reshape(X7.shape[0], img_rows, img_cols)
X8=X8.reshape(X8.shape[0], img_rows, img_cols)
X9=X9.reshape(X9.shape[0], img_rows, img_cols)
X10=X10.reshape(X10.shape[0], img_rows, img_cols)
#X11=X11.reshape(X11.shape[0], img_rows, img_cols)
#X12=X12.reshape(X12.shape[0], img_rows, img_cols)
#X13=X13.reshape(X13.shape[0], img_rows, img_cols)
#X14=X14.reshape(X14.shape[0], img_rows, img_cols)
#X15=X15.reshape(X15.shape[0], img_rows, img_cols)
#X16=X16.reshape(X16.shape[0], img_rows, img_cols)
#X17=X17.reshape(X17.shape[0], img_rows, img_cols)
#X18=X18.reshape(X18.shape[0], img_rows, img_cols)
#X19=X19.reshape(X19.shape[0], img_rows, img_cols)
#X20=X20.reshape(X20.shape[0], img_rows, img_cols)


#Plot an example image
fig = plt.figure()
plt.imshow(255-X3[0], cmap='gray', interpolation='none')
plt.xticks([])
plt.yticks([])

#Create x_train and y_train arrays
x_train=np.zeros((N*M,X1.shape[1],X1.shape[2]))
y_train=np.zeros((N*M,))
for i in range(N):    
    x_train[(i*M)+0,:,:]=X1[i,:,:]
    x_train[(i*M)+1,:,:]=X2[i,:,:]
    x_train[(i*M)+2,:,:]=X3[i,:,:]
    x_train[(i*M)+3,:,:]=X4[i,:,:]
    x_train[(i*M)+4,:,:]=X5[i,:,:]
    x_train[(i*M)+5,:,:]=X6[i,:,:]
    x_train[(i*M)+6,:,:]=X7[i,:,:]
    x_train[(i*M)+7,:,:]=X8[i,:,:]
    x_train[(i*M)+8,:,:]=X9[i,:,:]
    x_train[(i*M)+9,:,:]=X10[i,:,:]
#    x_train[(i*M)+10,:,:]=X11[i,:,:]
#    x_train[(i*M)+11,:,:]=X12[i,:,:]
#    x_train[(i*M)+12,:,:]=X13[i,:,:]
#    x_train[(i*M)+13,:,:]=X14[i,:,:]
#    x_train[(i*M)+14,:,:]=X15[i,:,:]
#    x_train[(i*M)+15,:,:]=X16[i,:,:]
#    x_train[(i*M)+16,:,:]=X17[i,:,:]
#    x_train[(i*M)+17,:,:]=X18[i,:,:]
#    x_train[(i*M)+18,:,:]=X19[i,:,:]
#    x_train[(i*M)+19,:,:]=X20[i,:,:]
    for j in range(M):
        y_train[(i*M)+j,]=j
    
y_train_labels=y_train


x_test=np.zeros((N_test*M,X1.shape[1],X1.shape[2]))
y_test=np.zeros((N_test*M,))
for i in range(N_test):    
    x_test[(i*M)+0,:,:]=X1[i+N,:,:]
    x_test[(i*M)+1,:,:]=X2[i+N,:,:]
    x_test[(i*M)+2,:,:]=X3[i+N,:,:]
    x_test[(i*M)+3,:,:]=X4[i+N,:,:]
    x_test[(i*M)+4,:,:]=X5[i+N,:,:]
    x_test[(i*M)+5,:,:]=X6[i+N,:,:]
    x_test[(i*M)+6,:,:]=X7[i+N,:,:]
    x_test[(i*M)+7,:,:]=X8[i+N,:,:]
    x_test[(i*M)+8,:,:]=X9[i+N,:,:]
    x_test[(i*M)+9,:,:]=X10[i+N,:,:]
#    x_test[(i*M)+10,:,:]=X11[i+N,:,:]
#    x_test[(i*M)+11,:,:]=X12[i+N,:,:]
#    x_test[(i*M)+12,:,:]=X13[i+N,:,:]
#    x_test[(i*M)+13,:,:]=X14[i+N,:,:]
#    x_test[(i*M)+14,:,:]=X15[i+N,:,:]
#    x_test[(i*M)+15,:,:]=X16[i+N,:,:]
#    x_test[(i*M)+16,:,:]=X17[i+N,:,:]
#    x_test[(i*M)+17,:,:]=X18[i+N,:,:]
#    x_test[(i*M)+18,:,:]=X19[i+N,:,:]
#    x_test[(i*M)+19,:,:]=X20[i+N,:,:]    
    for j in range(M):
        y_test[(i*M)+j,]=j
    
y_test_labels=y_test


#Para los triplets necesito un X_train con un índice más
x_train_ext=np.zeros((M,N,X1.shape[1],X1.shape[2]))
x_train_ext[0,:,:,:]=X1[0:N,:,:]
x_train_ext[1,:,:,:]=X2[0:N,:,:]
x_train_ext[2,:,:,:]=X3[0:N,:,:]
x_train_ext[3,:,:,:]=X4[0:N,:,:]
x_train_ext[4,:,:,:]=X5[0:N,:,:]
x_train_ext[5,:,:,:]=X6[0:N,:,:]
x_train_ext[6,:,:,:]=X7[0:N,:,:]
x_train_ext[7,:,:,:]=X8[0:N,:,:]
x_train_ext[8,:,:,:]=X9[0:N,:,:]
x_train_ext[9,:,:,:]=X10[0:N,:,:]
#x_train_ext[10,:,:,:]=X11[0:N,:,:]
#x_train_ext[11,:,:,:]=X12[0:N,:,:]
#x_train_ext[12,:,:,:]=X13[0:N,:,:]
#x_train_ext[13,:,:,:]=X14[0:N,:,:]
#x_train_ext[14,:,:,:]=X15[0:N,:,:]
#x_train_ext[15,:,:,:]=X16[0:N,:,:]
#x_train_ext[16,:,:,:]=X17[0:N,:,:]
#x_train_ext[17,:,:,:]=X18[0:N,:,:]
#x_train_ext[18,:,:,:]=X19[0:N,:,:]
#x_train_ext[19,:,:,:]=X20[0:N,:,:]




img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols) 
    x_train_ext = x_train_ext.reshape(x_train_ext.shape[0],N, 1, img_rows, img_cols)  
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_train_ext = x_train_ext.reshape(x_train_ext.shape[0],N, img_rows, img_cols,1)  
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
x_train_ext = x_train_ext.astype('float32')
x_train_ext /= 255

    

Triplets=True
if Triplets==True:          
    
    #Declare the model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(2, activation='linear'))
    
   
    
    #Declare the Siamese architecture
    Up_input = Input(shape = input_shape, name='Up_input')
    Down_input = Input(shape = input_shape, name='Down_input')
    Up_input2 = Input(shape = input_shape, name='Up_input2')
    Down_input2 = Input(shape = input_shape, name='Down_input2')    
    Label = Input(shape = (1,), name='label')
    
    Up_output = model(Up_input)
    Down_output = model(Down_input)
    Up_output2 = model(Up_input2)
    Down_output2 = model(Down_input2)    
    inputs_siamese = [Up_input, Down_input, Up_input2, Down_input2, Label]
    outputs_siamese = [Up_output, Down_output, Up_output2, Down_output2, Label]

        
    siamese_model = Model(inputs_siamese, outputs_siamese)
    siamese_model.add_loss(K.sum(contrastive_loss_triplet(outputs_siamese)))    
    siamese_model.compile(loss= None, optimizer='Adam', metrics=["accuracy"])
      
    
    #%% Train
    early_stopping_monitor = EarlyStopping(
        monitor='loss',
        min_delta=0,
        patience=10,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )
    
    batch_size = 256
    batches_per_epoch = 200 #500
    epochs = 10
    history_Motion = siamese_model.fit_generator(data_generator_clusters(x_train_ext, batch_size,N),
                                                          steps_per_epoch = batches_per_epoch,
                                                          epochs = epochs,                               
                                                          verbose = 1,
                                                          callbacks=[early_stopping_monitor ])
    
    
    #Plot the feature embedding
    trained_model = model
    X_train_trm = trained_model.predict(x_train[:1024].reshape(-1,28,28,1))
    scatter(X_train_trm, y_train[:1024], "Learned Feature Space",M)    
        
    
     
    
    
Softmax=False
if Softmax==True:    
        
    # convert class vectors to binary class matrices
    y_train_c = keras.utils.to_categorical(y_train, M)    
    y_test_c = keras.utils.to_categorical(y_test, M)
    
    #Declare the model
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='linear', name='feature_layer'))
    model.add(Dense(M, activation='softmax'))


    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    
    
    early_stopping_monitor = EarlyStopping(
        monitor='accuracy',
        min_delta=0,
        patience=15,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True
    )
    
    
    batch_size=256
    model.fit(x_train, y_train_c,
              batch_size=batch_size,
              epochs=200,
              verbose=1,
              callbacks=[early_stopping_monitor],
              validation_data=(x_test, y_test_c))
    
    
    #Select the layer with two units
    embedding_layer = model.get_layer("feature_layer").output            
    feature_extractor = Model(model.input, embedding_layer)
            
       
    #Plot the feature embedding
    trained_model=feature_extractor    
    X_train_trm = trained_model.predict(x_train[:1024].reshape(-1,28,28,1))
    scatter(X_train_trm, y_train_labels[:1024], "Learned Feature Space",M)    
    
    



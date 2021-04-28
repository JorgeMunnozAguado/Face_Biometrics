# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 21:01:47 2020

@author: aytha
"""
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

M=20 #number of classes
N=100 #number of samples per class

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
X11=np.load('full_numpy_bitmap_asparagus.npy')
X12=np.load('full_numpy_bitmap_axe.npy')
X13=np.load('full_numpy_bitmap_backpack.npy')
X14=np.load('full_numpy_bitmap_banana.npy')
X15=np.load('full_numpy_bitmap_bandage.npy')
X16=np.load('full_numpy_bitmap_The Great Wall of China.npy')
X17=np.load('full_numpy_bitmap_The Eiffel Tower.npy')
X18=np.load('full_numpy_bitmap_book.npy')
X19=np.load('full_numpy_bitmap_barn.npy')
X20=np.load('full_numpy_bitmap_bird.npy')

#Rechape images into 2D space
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
X11=X11.reshape(X11.shape[0], img_rows, img_cols)
X12=X12.reshape(X12.shape[0], img_rows, img_cols)
X13=X13.reshape(X13.shape[0], img_rows, img_cols)
X14=X14.reshape(X14.shape[0], img_rows, img_cols)
X15=X15.reshape(X15.shape[0], img_rows, img_cols)
X16=X16.reshape(X16.shape[0], img_rows, img_cols)
X17=X17.reshape(X17.shape[0], img_rows, img_cols)
X18=X18.reshape(X18.shape[0], img_rows, img_cols)
X19=X19.reshape(X19.shape[0], img_rows, img_cols)
X20=X20.reshape(X20.shape[0], img_rows, img_cols)


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
    x_train[(i*M)+10,:,:]=X11[i,:,:]
    x_train[(i*M)+11,:,:]=X12[i,:,:]
    x_train[(i*M)+12,:,:]=X13[i,:,:]
    x_train[(i*M)+13,:,:]=X14[i,:,:]
    x_train[(i*M)+14,:,:]=X15[i,:,:]
    x_train[(i*M)+15,:,:]=X16[i,:,:]
    x_train[(i*M)+16,:,:]=X17[i,:,:]
    x_train[(i*M)+17,:,:]=X18[i,:,:]
    x_train[(i*M)+18,:,:]=X19[i,:,:]
    x_train[(i*M)+19,:,:]=X20[i,:,:]
    for j in range(M):
        y_train[(i*M)+j,]=j
    
x_train = x_train.astype('float32')
x_train /= 255  

    
    

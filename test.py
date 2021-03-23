
import os
import cv2
import numpy as np

from scipy import misc

from keras_vggface.vggface import VGGFace
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Lambda, Activation, ActivityRegularization



classes = {'HA':0, 'HB':1, 'HN':2, 'MA':3, 'MB':4, 'MN':5}



def loadImage(path, image_size=224):

    path = os.path.join(path, os.listdir(path)[0])

    # Read image
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_LINEAR)

    # Prepare for inference
    img = np.expand_dims(img, axis=0)


    return img, path





def loadSet(path, model):


    # List sets
    clss = os.listdir(path)
    file = open(os.path.join(path, 'embeddings.csv'), 'w')

    file.write("file_name,label,embedding\n")

    # For each set
    for c in clss:

        # Class of set
        key = c.split('4K_120')[0]

        # List subjects
        group_path = os.path.join(path, c)
        data = os.listdir(group_path)

        # print(key, classes[key])
        

        for name in data:

            img, file_name = loadImage( os.path.join(group_path, name) )

            # Inference model
            embedding = model.predict(img)

            file.write("{},{},{}\n".format(file_name, classes[key], embedding))


    file.close()



def loadModel(verbose=0):

    # Import the ResNet-50 model trained with VGG2 database
    my_model = 'resnet50'
    resnet = VGGFace(model = my_model)
    
    if verbose:  resnet.summary()

    # Select the lat leayer as feature embedding  
    last_layer = resnet.get_layer('avg_pool').output
    feature_layer = Flatten(name='flatten')(last_layer)
    model_vgg = Model(resnet.input, feature_layer)

    return model_vgg



if __name__ == '__main__':

    model = loadModel()
    loadSet('4K_120', model)
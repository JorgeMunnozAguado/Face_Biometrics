
import os
import cv2
import numpy as np

from scipy import misc

from keras_vggface.vggface import VGGFace
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout, Flatten, Lambda, Activation, ActivityRegularization

from dataset import classes



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



def printArray(file, array):

    for a in array:

        file.write(",{}".format(a))



def loadSet(path, model, cut=50, verbose=0):


    # List sets
    clss = os.listdir(path)
    clss = [ cl for cl in clss if os.path.isdir(os.path.join(path, cl)) ]


    # Open csv file
    file = open(os.path.join(path, 'embeddings.csv'), 'w')

    # file.write("file_name,label,embedding\n")
    file.write("file_name,label")

    header = ['embedding%d'%i for i in range(2048)]
    printArray(file, header)

    file.write("\n")



    # For each set
    for c in clss:

        # Class of set
        key = c.split('4K_120')[0]

        # List subjects
        group_path = os.path.join(path, c)
        data = os.listdir(group_path)[:cut]

        if verbose:  print('Class', c,'   - len:', len(data))


        for idx, name in enumerate(data):

            if verbose > 1:  print(idx)

            img, file_name = loadImage( os.path.join(group_path, name) )

            # Inference model
            embedding = model.predict(img)

            # file.write("{},{},{}\n".format(file_name, classes[key], embedding.tolist()))
            file.write("{},{}".format(file_name, classes[key]))

            printArray(file, embedding[0])

            file.write("\n")

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
    loadSet('4K_120', model, cut=150, verbose=1)

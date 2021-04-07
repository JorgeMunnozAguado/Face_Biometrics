
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.utils import to_categorical

from dataset import loadData, changeGroups, dtypes
from dataset import classes, classes_race, classes_gender

def defineModel(input_shape, num_outputs):

    nn = Sequential()

    nn.add(Dense(1000, input_shape=input_shape))
    nn.add(Dense(100, activation='sigmoid'))
    nn.add(Dense(num_outputs, activation='softmax'))

    nn.compile(optimizer=Adam(learning_rate=0.001), loss=CategoricalCrossentropy(), metrics='accuracy')

    return nn


def plotHistory(history, title):

    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Accuracy  - ' + title)
    plt.ylabel('')
    plt.xlabel('Epoch')
    plt.legend(loc="upper left")
    plt.show()

    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Binary Crossentropy  - ' + title)
    plt.ylabel('')
    plt.xlabel('Epoch')
    plt.legend(loc="upper left")
    plt.show()


def test_model(model, embeddings, labels, pre_data=True):

    if pre_data:

        labels = to_categorical(labels)

        embeddings = np.expand_dims(embeddings, axis=0)
        labels     = np.expand_dims(labels, axis=0)


    # Fit model to data
    history = model_race.fit(embeddings, labels_race, batch_size=batch_size, epochs=15, verbose=1, validation_split=0.2)
    #history = model_race.fit(embeddings, labels_race, batch_size=batch_size, epochs=50, verbose=verbose)

    return model, history



def classifier(csv_file, need_classes, title, batch_size=30, plot=True, verbose=0):

    # Load data
    embeddings, labels = loadData(csv_file, dtypes)

    # Prepare data
    labels = changeGroups(labels, classes, need_classes)
    labels = to_categorical(labels)

    embeddings = np.expand_dims(embeddings, axis=0)
    labels     = np.expand_dims(labels, axis=0)

    output_size = len(np.unique( list(need_classes.values()) ))
    print('----------------------------->', output_size, title)

    # Define model
    model = defineModel(embeddings.shape[1:], output_size)
    if verbose:  model_race.summary()


    # Fit model to data
    history = model.fit(embeddings, labels, batch_size=batch_size, epochs=15, verbose=1, validation_split=0.2)


    if plot:  plotHistory(history, title)





if __name__ == '__main__':

    csv_file = '4K_120/embeddings.csv'

    classifier(csv_file, classes_race, 'race')
    classifier(csv_file, classes_gender, 'gender')

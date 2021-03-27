
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.utils import to_categorical

from t_SNE import loadData, changeGroups
from embeddings import classes

def defineModel(input_shape, num_outputs):

    nn = Sequential()

    nn.add(Dense(1000, input_shape=input_shape))
    nn.add(Dense(100, activation='sigmoid'))
    nn.add(Dense(num_outputs, activation='softmax'))

    nn.compile(optimizer=Adam(learning_rate=0.001), loss=CategoricalCrossentropy(), metrics='accuracy')

    return nn


def plotHistory(history):

    plt.plot(history.history['accuracy'], label='Accuracy')
    # plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.title('Accuracy')
    plt.ylabel('')
    plt.xlabel('Epoch')
    plt.legend(loc="upper left")
    plt.show()

    plt.plot(history.history['loss'], label='Loss')
    # plt.plot(history.history['val_loss'], label='Validation loss')
    plt.title('Binary Crossentropy')
    plt.ylabel('')
    plt.xlabel('Epoch')
    plt.legend(loc="upper left")
    plt.show()




def raceClassifier(csv_file, dtypes, batch_size=30, plot=True, verbose=0):

    groups_race = {'HA':0, 'HB':1, 'HN':2, 'MA':0, 'MB':1, 'MN':2}


    # Prepare data
    embeddings, labels = loadData(csv_file, dtypes)

    labels_race = changeGroups(labels, classes, groups_race)
    labels_race = to_categorical(labels_race)

    embeddings  = np.expand_dims(embeddings, axis=0)
    labels_race = np.expand_dims(labels_race, axis=0)


    # Define model
    model_race = defineModel(embeddings.shape[1:], 3)

    if verbose:  model_race.summary()


    # Fit model to data
    # history = model_race.fit(embeddings, labels_race, epochs=15, verbose=1, validation_split=0.2)
    history = model_race.fit(embeddings, labels_race, batch_size=batch_size, epochs=50, verbose=verbose)

    if plot:  plotHistory(history)



def genderClassifier(csv_file, dtypes, batch_size=30, plot=True, verbose=0):

    groups_gender = {'HA':0, 'HB':0, 'HN':0, 'MA':1, 'MB':1, 'MN':1}


    # Prepare data
    embeddings, labels = loadData(csv_file, dtypes)

    labels_race = changeGroups(labels, classes, groups_gender)
    labels_race = to_categorical(labels_race)

    embeddings  = np.expand_dims(embeddings, axis=0)
    labels_race = np.expand_dims(labels_race, axis=0)


    # Define model
    model_race = defineModel(embeddings.shape[1:], 2)

    if verbose:  model_race.summary()


    # Fit model to data
    # history = model_race.fit(embeddings, labels_race, epochs=15, verbose=1, validation_split=0.2)
    history = model_race.fit(embeddings, labels_race, batch_size=batch_size, epochs=50, verbose=verbose)

    if plot:  plotHistory(history)



if __name__ == '__main__':

    csv_file = '4K_120/embeddings.csv'

    dtypes = {'file_name' : str,
              'label' : int,
              'embedding' : float}



    # raceClassifier(csv_file, dtypes)
    genderClassifier(csv_file, dtypes)


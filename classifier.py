
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.losses import CategoricalCrossentropy
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

from keras_vggface.vggface import VGGFace

from dataset import loadData, changeGroups, splitGroups, onlyGroup, searchDict, dtypes
from dataset import classes, classes_race, classes_gender

def defineModel(input_shape, num_outputs):

    nn = Sequential()

    nn.add(Dense(1000, input_shape=input_shape))
    nn.add(Dense(100, activation='sigmoid'))
    nn.add(Dense(num_outputs, activation='softmax'))

    nn.compile(optimizer=Adam(learning_rate=0.001), loss=CategoricalCrossentropy(), metrics='accuracy')

    return nn

def completeModel(num_outputs):

    resnet = VGGFace(model='resnet50')

    if verbose:  resnet.summary()

    # Select the lat leayer as feature embedding
    last_layer = resnet.get_layer('avg_pool').output
    feature_layer = Flatten(name='flatten')(last_layer)
    model_vgg = Model(resnet.input, feature_layer)



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

    val = history.history['val_accuracy']
    idx = np.argmax(val)

    print('Results max (' + title + '):   val acc: %f  (epoch: %d)'%(val[idx], idx))



def simple_classification(csv_file, need_classes, title, batch_size=30, plot=True, verbose=0):

    # Load data
    embeddings, labels = loadData(csv_file, dtypes)

    # Prepare data
    labels = changeGroups(labels, classes, need_classes)
    labels = to_categorical(labels)

    output_size = len(np.unique( list(need_classes.values()) ))


    # Split data
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.33)


    classifier(X_train, X_test, y_train, y_test, output_size, title, batch_size=batch_size, plot=plot, verbose=verbose)



def different_classification(csv_file, train_dict, need_classes, batch_size=30, plot=True, verbose=0):
    '''
    train_dict = {0: for test, 1 for training}
    '''

    # Load data
    embeddings, labels = loadData(csv_file, dtypes)

    # Prepare data
    X_train, X_test, y_train, y_test  = splitGroups(embeddings, labels, need_classes, classes, train_dict)

    y_train = to_categorical(y_train)
    y_test  = to_categorical(y_test, 3)


    output_size = len(np.unique( list(need_classes.values()) ))


    classifier(X_train, X_test, y_train, y_test, output_size, 'test', batch_size=batch_size, plot=plot, verbose=verbose)





def classifier(X_train, X_test, y_train, y_test, output_size, title, batch_size=30, plot=True, verbose=0):


    # Prepare data
    X_train = np.expand_dims(X_train, axis=0)
    X_test  = np.expand_dims(X_test, axis=0)
    y_train = np.expand_dims(y_train, axis=0)
    y_test  = np.expand_dims(y_test, axis=0)


    # Define model
    model = defineModel(X_train.shape[1:], output_size)
    if verbose:  model.summary()


    # Fit model to data
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=50, validation_data=(X_test, y_test), verbose=verbose)


    # Plot model
    if plot:  plotHistory(history, title)


    return model, history



def train(X_train, y_train, output_size, epochs=50, batch_size=30, verbose=0):


    # Prepare data
    X_train = np.expand_dims(X_train, axis=0)
    y_train = np.expand_dims(y_train, axis=0)

    shape = (None, X_train.shape[2], )

    # Define model
    model = defineModel(shape, output_size)
    if verbose:  model.summary()


    # Fit model to data
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose)

    return model, history



def evaluate(X_test, y_test, model, verbose=0):

    # Prepare data
    X_test  = np.expand_dims(X_test, axis=0)
    y_test  = np.expand_dims(y_test, axis=0)

    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=verbose)

    return loss, accuracy


def saveModel(model, filename='model_checkpoint'):

    model.save_weights('checkpoints/' + filename)


def loadModel(model, filename):

    model.load_weights('checkpoints/' + filename).expect_partial()

    return model


def split_classes(csv_file):

    embeddings, labels = loadData(csv_file, dtypes)

    data_dict = {}


    for cl, id in classes.items():

        
        embeddings_cl, labels_cl = onlyGroup(embeddings, labels, [id], classes)

        X_train, X_test, y_train, y_test = train_test_split(embeddings_cl, labels_cl, test_size=250)

        data_dict[cl] = {'train': [X_train, y_train], 'test': [X_test, y_test]}


    return data_dict


if __name__ == '__main__':

    csv_file = '4K_120/embeddings.csv'

    # simple_classification(csv_file, classes_race, 'race')
    # simple_classification(csv_file, classes_gender, 'gender')


    # Ej. 4
    train_dict = {'HA':1, 'HB':1, 'HN':1, 'MA':0, 'MB':0, 'MN':0}

    different_classification(csv_file, train_dict, classes_race)

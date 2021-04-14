
import random
import numpy as np
import pandas as pd

classes = {'HA':0, 'HB':1, 'HN':2, 'MA':3, 'MB':4, 'MN':5}
classes_race = {'HA':0, 'HB':1, 'HN':2, 'MA':0, 'MB':1, 'MN':2}
classes_gender = {'HA':0, 'HB':0, 'HN':0, 'MA':1, 'MB':1, 'MN':1}


label = ['Hombre Asiático', 'Hombre Blanco', 'Hombre Negro', 'Mujer Asiática', 'Mujer Blanca', 'Mujer Negra']
label_race = ['Asiatico', 'Blanco', 'Negro']
label_gender = ['Hombre', 'Mujer']

dtypes = {'file_name' : str,
          'label' : int,
          'embedding' : float}


def searchDict(value, dict):

    lista = []

    for key, val in dict.items():

        if val == value: lista.append(key)

    return lista


def suffle_array(X, y):

    c = list(zip(X, y))

    random.shuffle(c)

    X, y = zip(*c)
    X, y = np.asarray(X), np.asarray(y)


    return X, y




def loadData(csv_file, dtypes=dtypes, suffle=True):

    data = pd.read_csv(csv_file, dtype=dtypes)

    embeddings = data.iloc[:, 2:]
    embeddings = embeddings.to_numpy()

    labels = data.iloc[:, 1]
    labels = labels.to_numpy()

    # Suffle
    if suffle:
        embeddings, labels = suffle_array(embeddings, labels)


    return embeddings, labels



def changeGroups(labels, classes, groups):

    labels = labels.copy()

    for cl, id in classes.items():

        new_id = groups[cl]

        idx = np.where(labels == id)
        labels[idx] = new_id

    return labels


def splitGroups(embeddings, labels, need_classes, classes, groups, suffle=True):
    '''
    groups = {0: for test, 1 for training}
    '''

    X_train, X_test, y_train, y_test = [], [], [], []



    for cl, id in classes.items():

        new_id = groups[cl]

        # print(id, new_id)

        idx = np.where(labels == id)[0]

        aux = labels[idx]
        aux = np.full(aux.shape, need_classes[cl])

        if new_id == 0:

            X_test += list(embeddings[idx])
            y_test += list(aux)


        elif new_id == 1:

            X_train += list(embeddings[idx])
            y_train += list(aux)


    # Suffle
    if suffle:
        X_train, y_train = suffle_array(X_train, y_train)
        X_test, y_test   = suffle_array(X_test, y_test)


    return X_train, X_test, y_train, y_test


def onlyGroup(embeddings, labels, need_classes, classes):

    X, y = [], []

    for id in need_classes:

        idx = np.where(labels == id)[0]

        X += list(embeddings[idx])
        y += list(labels[idx])


    return np.asarray(X), np.asarray(y)

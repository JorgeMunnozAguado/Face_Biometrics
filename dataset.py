
import numpy as np
import pandas as pd

classes = {'HA':0, 'HB':1, 'HN':2, 'MA':3, 'MB':4, 'MN':5}
classes_race = {'HA':0, 'HB':1, 'HN':2, 'MA':0, 'MB':1, 'MN':2}
classes_gender = {'HA':0, 'HB':0, 'HN':0, 'MA':1, 'MB':1, 'MN':1}


label = ['Hombre Asiático', 'Hombre Blanco', 'Hombre Negro', 'Mujer Asiática', 'Mujer Blanca', 'Mujer Negra']
label_race = ['Asiático', 'Blanco', 'Negro']
label_gender = ['Hombre', 'Mujer']


dtypes = {'file_name' : str,
          'label' : int,
          'embedding' : float}


def loadData(csv_file, dtypes=dtypes):

    data = pd.read_csv(csv_file, dtype=dtypes)

    embeddings = data.iloc[:, 2:]
    embeddings = embeddings.to_numpy()

    labels = data.iloc[:, 1]
    labels = labels.to_numpy()

    return embeddings, labels



def changeGroups(labels, classes, groups):

    labels = labels.copy()

    for cl, id in classes.items():

        new_id = groups[cl]

        idx = np.where(labels == id)
        labels[idx] = new_id

    return labels

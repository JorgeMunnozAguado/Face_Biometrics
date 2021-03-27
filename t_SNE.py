
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from embeddings import classes



def loadData(csv_file, dtypes):

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


def calculateTSNE(embeddings, labels, n_components, label_list=None, display=False):

    repre = TSNE(n_components=n_components, random_state=0).fit_transform(embeddings)

    if display:

        fig = plt.figure(figsize = (10, 10))
        ax  = fig.add_subplot(111)

        scatter = ax.scatter(repre[:, 0], repre[:, 1], c=labels, cmap='tab10')
        handles, labels = scatter.legend_elements()

        if label_list:  labels = label_list

        legend = ax.legend(handles=handles, labels=labels)

        plt.show()




if __name__ == '__main__':

    csv_file = '4K_120/embeddings.csv'

    dtypes = {'file_name' : str,
              'label' : int,
              'embedding' : float}

    n_components = 2
    groups_1 = {'HA':0, 'HB':1, 'HN':2, 'MA':0, 'MB':1, 'MN':2}
    groups_2 = {'HA':0, 'HB':0, 'HN':0, 'MA':1, 'MB':1, 'MN':1}
    label_list_1 = ['Asiático', 'Blanco', 'Negro']
    label_list_2 = ['Hombre', 'Mujer']


    # Load data
    embeddings, labels = loadData(csv_file, dtypes)

    labels_1 = changeGroups(labels, classes, groups_1)
    labels_2 = changeGroups(labels, classes, groups_2)

    calculateTSNE(embeddings, labels_1, n_components, label_list=label_list_1, display=True)
    calculateTSNE(embeddings, labels_2, n_components, label_list=label_list_2, display=True)
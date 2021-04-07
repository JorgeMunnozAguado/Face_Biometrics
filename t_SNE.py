
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

from dataset import classes, classes_race, classes_gender, label, label_race, label_gender, dtypes
from dataset import loadData, changeGroups


def calculateTSNE(embeddings, labels, n_components=2, label_list=None, display=False, title=None):

    repre = TSNE(n_components=n_components, random_state=0).fit_transform(embeddings)

    if display:

        fig = plt.figure(figsize = (10, 10))
        ax  = fig.add_subplot(111)

        scatter = ax.scatter(repre[:, 0], repre[:, 1], c=labels, cmap='tab10')
        handles, labels = scatter.legend_elements()

        if label_list:  labels = label_list

        legend = ax.legend(handles=handles, labels=labels)

        if title:  plt.title(title)
        plt.show()

    return repre




if __name__ == '__main__':

    csv_file = '4K_120/embeddings.csv'


    # Load data
    embeddings, labels = loadData(csv_file, dtypes)

    labels_race = changeGroups(labels, classes, classes_race)
    labels_gender = changeGroups(labels, classes, classes_gender)

    calculateTSNE(embeddings, labels_race, label_list=label_race, display=True, title='race')
    calculateTSNE(embeddings, labels_gender, label_list=label_gender, display=True, title='gender')
    calculateTSNE(embeddings, labels, label_list=label, display=True, title='All')



from keras.models import Sequential
from keras.layers import Dense, AveragePooling2D

from t_SNE import loadData


def defineModel(num_outputs):

    nn = Sequential()

    nn.add(AveragePooling2D())
    nn.add(Dense(2, activation=None))


    # TODO -   ADAM
    nn.compile(optimizer="SGD", loss='mean_squared_error', metrics='accuracy')



if __name__ == '__main__':

    csv_file = '4K_120/embeddings.csv'

    dtypes = {'file_name' : str,
              'label' : int,
              'embedding' : float}

    # groups = {'HA':0, 'HB':1, 'HN':2, 'MA':0, 'MB':1, 'MN':2}


    embeddings, labels = loadData(csv_file, dtypes)

    # labels = changeGroups(labels, classes, groups)
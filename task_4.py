
import numpy as np
from keras.utils import to_categorical


from classifier import split_classes, train, evaluate
from dataset import suffle_array, label_race


def dict2list(data_dict, classes, type, suffle=True):

    X, y = [], []

    for i, cl in enumerate(classes):

        embedding = data_dict[cl][type][0] 

        X += list( embedding )

        # ONE HOT ENCODER
        one_hot_label = np.zeros((len(embedding), len(classes)))
        one_hot_label[:, i] = 1

        y += list( one_hot_label )


    if suffle:   X, y = suffle_array(X, y)


    return np.asarray(X), np.asarray(y)




def train_test_group(data_dict, classes, title):

    X_train, y_train = dict2list(data_dict, classes, 'train')


    model, history = train(X_train, y_train, 2)


    X_test_asia, y_test_asia = dict2list(data_dict, ['HA', 'MA'], 'test')
    X_test_blan, y_test_blan = dict2list(data_dict, ['HB', 'MB'], 'test')
    X_test_negr, y_test_negr = dict2list(data_dict, ['HN', 'MN'], 'test')


    loss_asia, accuracy_asia = evaluate(X_test_asia, y_test_asia, model)
    loss_blan, accuracy_blan = evaluate(X_test_blan, y_test_blan, model)
    loss_negr, accuracy_negr = evaluate(X_test_negr, y_test_negr, model)


    print("Model trained with", title + 's       accuracy (loss):')

    print("  -> test asiaticos: %.2f (%.4f)" % (accuracy_asia, loss_asia))
    print("  -> test blancos: %.2f (%.4f)" % (accuracy_blan, loss_blan))
    print("  -> test negros: %.2f (%.4f)" % (accuracy_negr, loss_negr))

    print('---------------------------------------------\n\n')



if __name__ == '__main__':

    csv_file = '4K_120/embeddings.csv'

    data_dict = split_classes(csv_file)

    print()

    train_test_group(data_dict, ['HA', 'MA'], label_race[0])
    train_test_group(data_dict, ['HB', 'MB'], label_race[1])
    train_test_group(data_dict, ['HN', 'MN'], label_race[2])

    # train_test_group(data_dict, ['HA', 'MA', 'HB', 'MB', 'HN', 'MN'], label_race[0])

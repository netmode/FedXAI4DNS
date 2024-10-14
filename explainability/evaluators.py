''' Python Non-Standard Libraries '''
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Input, Dropout
from tensorflow.keras import callbacks

''' Python Local Libraries '''
from helpers import debug
from env import PARTICIPANTS


def evaluate_nonfed_model(model, X_test, y_test):
    ''' Calculates the accuracy of the model on the testing set '''

    y_test_temp = y_test.iloc[:, 1]
    score = model.evaluate(X_test, y_test_temp, verbose = 1)

    filename = 'accuracy_results/nonfed_model_accuracy.txt'
    fdw = open(filename, 'w')

    fdw.write('Model Loss:' + str(score[0]) + '\n')
    fdw.write('Model Accuracy:' + str(score[1]) + '\n')
    fdw.close()

    return None


def evaluate_nonfed_model_clients(local_models, X_test, y_test, filename):
    ''' Calculates the accuracy on the local  models trained in a
        non federated manner
    '''

    y_test_temp = y_test.iloc[:, 1]
    path = 'accuracy_results/' + filename
    text = 'Accuracy of participant {} is {}\n'
    fdw = open(path, 'w')

    for participant in range(PARTICIPANTS):
        model = local_models[participant]
        score = model.evaluate(X_test, y_test_temp, verbose = 1)
        to_write = text.format(
                str(participant), str(score[1])
                )
        fdw.write(to_write)

    fdw.close()

    return None


def evaluate_fed_model(local_models, global_model, X_test, y_test, filename):
    ''' Calculates the accuracy on the local and global models of the 
        Federated Learning case
    '''

    y_test_temp = y_test.iloc[:, 1]
    path = 'accuracy_results/' + filename
    text = 'Accuracy of participant {} is {}\n'
    fdw = open(path, 'w')

    for participant in range(PARTICIPANTS):
        model = local_models[participant]
        score = model.evaluate(X_test, y_test_temp, verbose = 1)
        to_write = text.format(
                str(participant), str(score[1])
                )
        fdw.write(to_write)

    global_score = global_model.evaluate(X_test, y_test_temp, verbose = 1)
    to_write = text.format(
            'global', str(global_score[1])
            )
    fdw.write(to_write)

    fdw.close()

    return None

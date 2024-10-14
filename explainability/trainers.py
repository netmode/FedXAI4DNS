''' Python Non-Standard Libraries '''
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Input, Dropout
from tensorflow.keras import callbacks
import numpy as np

''' Python Local Libraries '''
from env import FED_ROUNDS, FED_EPOCHS, NON_FED_EPOCHS, PARTICIPANTS
from helpers import debug


def define_model(features_number):
    ''' Defines the MLP model '''

    model = tf.keras.models.Sequential()
    model.add(
            tf.keras.layers.Dense(
                300, input_dim = features_number, activation = 'relu'
                )
            )
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(300, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(200, activation = 'relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation = 'sigmoid'))

    model.compile(
            loss = 'binary_crossentropy', optimizer = 'adam',\
                    metrics = ['accuracy'])

    return model


def non_federated_trainer(X_train, y_train):
    ''' Train models in a non federated manner '''

    early_stopping = callbacks.EarlyStopping(
            monitor = 'val_loss', patience = 5, verbose = 1
            )

    model = define_model(len(X_train.columns))

    history = model.fit(
            X_train, y_train.values.ravel(),\
                    validation_split = 0.2,\
                    epochs = NON_FED_EPOCHS,\
                    batch_size = 256,\
                    callbacks = [early_stopping]
            )

    return model


def nonfed_local_model_trainer(X_train_list, y_train_list):
    ''' Train multiple local models in a non federated manner
        in one function
    '''

    local_models = [None] * PARTICIPANTS

    for participant in range(PARTICIPANTS):
        X_train = X_train_list[participant]
        y_train = y_train_list[participant]
        local_model = non_federated_trainer(X_train, y_train)
        local_models[participant] = local_model

    return local_models


def train_client_model(model, X_train, y_train):
    ''' Train the client model during a federated round '''

    history = model.fit(
            X_train, y_train.values.ravel(),\
                    validation_split = 0.2,\
                    epochs = FED_EPOCHS,\
                    batch_size = 256
            )

    return model


def aggregate_model_weights(models):
    ''' Aggregates local models to the global one '''

    for participant in range(PARTICIPANTS):
        debug('Inside aggregate', models[participant].get_weights())

    global_weights = []
    for weights_list in zip(*[model.get_weights() for model in models]):
        global_weights.append(np.mean(weights_list, axis = 0))

    debug('Global weights:', global_weights)

    return global_weights


def federated_trainer(X_train_list, y_train_list):
    ''' Trains the Federated Model '''

    debug('Training in a federated manner')

    features_number = len(X_train_list[0].columns)
    global_model = define_model(features_number)

    local_models = [None] * PARTICIPANTS
    for participant in range(PARTICIPANTS):
        local_models[participant] = define_model(features_number)

    for round_num in range(FED_ROUNDS):
        debug('Federated Round:', round_num)

        for participant in range(PARTICIPANTS):
            if round_num > 0:
                local_models[participant].set_weights(global_weights)
                debug('Local model weights before federated training:',\
                        local_models[participant].get_weights())
            local_models[participant] = train_client_model(
                    local_models[participant],
                    X_train_list[participant],
                    y_train_list[participant]
                    )
            debug('Local model weights:', local_models[participant].get_weights())

        global_weights = aggregate_model_weights(local_models)
        global_model.set_weights(global_weights)

    return local_models, global_model

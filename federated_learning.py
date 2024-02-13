import os
import math

import pickle

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import keras
from tensorflow.keras.models import save_model, load_model

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import shap


def create_train_test_sets(train_df1, train_df2, train_df3, train_df4, train_df5, test_df):
    '''
        Create train and test datasets in a supervised format (X, y)

        Input:
            train and test dataframes

        Return:
            X,y for every dataframe
    '''
    # Drop domain name, family and Label column to create X_train data.
    X_train_1 = np.array(train_df1.drop(["Domain Name", "Family", "Label"], axis=1))

    # Create a y np array with the labels.
    y_train_1 = np.array(train_df1["Label"])

    # Drop domain name, family and Label column to create X_train data.
    X_train_2 = np.array(train_df2.drop(["Domain Name", "Family", "Label"], axis=1))

    # Create a y np array with the labels.
    y_train_2 = np.array(train_df2["Label"])

    # Drop domain name, family and Label column to create X_train data.
    X_train_3 = np.array(train_df3.drop(["Domain Name", "Family", "Label"], axis=1))

    # Create a y np array with the labels.
    y_train_3 = np.array(train_df3["Label"])

    # Drop domain name, family and Label column to create X_train data.
    X_train_4 = np.array(train_df4.drop(["Domain Name", "Family", "Label"], axis=1))

    # Create a y np array with the labels.
    y_train_4 = np.array(train_df4["Label"])

    # Drop domain name, family and Label column to create X_train data.
    X_train_5 = np.array(train_df5.drop(["Domain Name", "Family", "Label"], axis=1))

    # Create a y np array with the labels.
    y_train_5 = np.array(train_df5["Label"])

    # Drop domain name, family and Label column to create X_train data.
    X_test = np.array(test_df.drop(["Domain Name", "Family", "Label"], axis=1))

    # Create a y np array with the labels.
    y_test = np.array(test_df["Label"])

    return X_train_1, y_train_1, X_train_2, y_train_2, X_train_3, y_train_3, X_train_4, y_train_4, X_train_5, y_train_5, X_test, y_test


def MinMax_normalization(X_train, y_train, X_test, y_test):
    '''
        Normalize data using MinMax Normalization

            Input:
                Train and test set

            Return:
                Scaled train, validation and test set
    '''

    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.2, shuffle=True,stratify=y_train, random_state=42)

    # Create a scaler based on train dataset.
    scaler_obj = MinMaxScaler()
    X_train_scaled = scaler_obj.fit_transform(X_train)

    # Transform validation and test sety based on the training scaler.
    X_validation_scaled = scaler_obj.transform(X_validation)

    X_test_scaled = scaler_obj.transform(X_test)

    return X_train_scaled, y_train, X_validation_scaled, y_validation, X_test_scaled, y_test


def create_model(n_features):
    '''
        Create an MLP model.
    '''
    model = Sequential([
        Dense(units=128, input_shape=(n_features,), activation='relu'),
        #         Dropout(rate = 0.2),
        Dense(units=32, activation='relu'),
        #         Dropout(rate = 0.2),
        Dense(units=16, activation='relu'),
        #         Dropout(rate = 0.2),
        Dense(units=1, activation='sigmoid')
    ])

    # Compile model.
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train_model(model, X_train, y_train, X_validation, y_validation):
    '''
        Train model for 1 epoch with batch size 32

        Input:
            model: the model you want to train
            train and validation sets

        Return:
            the weights of the model after training and training and validation loss
    '''
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_validation, y_validation),verbose=0)

    model_current_weights = model.get_weights()

    return model, model_current_weights, history.history['loss'][0], history.history['val_loss'][0]


def evaluate_model(model, X_test, y_test):
    '''
        Evaluate model in test set

        Input:
            model: the model you want to evaluate
            X,y test sets

        Print:
            Accuracy, precision, recall and f1-score
    '''
    # Get the predictions of the model.
    y_pred = model.predict(X_test)
    y_pred = y_pred.flatten()
    y_pred = np.round(y_pred)
    y_pred = y_pred.astype(int)

    print('Accuracy: {}\nPrecision: {}\nRecall: {}\nF1-Score: {}'.format(accuracy_score(y_test, y_pred),precision_score(y_test, y_pred),recall_score(y_test, y_pred),f1_score(y_test, y_pred)))


def training_results_line_plot(train_loss_lst, validation_loss_lst):
    '''
    '''
    # Create a figure with a 3x3 grid of subplots
    fig = plt.figure(figsize=(10, 5))  # You can adjust the figsize as needed

    # Create subplots with two lines in each subplot
    plt.plot(train_loss_lst, label='Training Loss')
    plt.plot(validation_loss_lst, label='Validation Loss')
    plt.title('Train Results')
    plt.legend()

    # Adjust the layout and spacing
    plt.tight_layout()

    # plt.savefig('lstm_24_loss_plot.png')

    # Show the plots
    plt.show()

def train_client_model(model_name, n_features, X_train, y_train, X_validation, y_validation, X_test, y_test, save_path):
    print('Train {}'.format(model_name))

    # Create model
    train_loss_lst = []
    validation_loss_lst = []
    model = create_model(n_features)

    # Train model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                        validation_data=(X_validation, y_validation), verbose=0)

    train_loss_lst.append(history.history['loss'])
    validation_loss_lst.append(history.history['val_loss'])

    # Evaluate model to the test set.
    evaluate_model(model, X_test, y_test)

    # Save the model
    model.save(save_path + '/' + model_name + '.h5')

if __name__ == '__main__':

    # Load Dataset.
    train_df1 = pd.read_csv("C:/Nikolaos Sintoris/01) Education/Msc DSML  - NTUA\Thesis/data/v4/high_heterogeneous_v4/high_heterogeneous_df1.csv")
    train_df2 = pd.read_csv("C:/Nikolaos Sintoris/01) Education/Msc DSML  - NTUA\Thesis/data/v4/high_heterogeneous_v4/high_heterogeneous_df2.csv")
    train_df3 = pd.read_csv("C:/Nikolaos Sintoris/01) Education/Msc DSML  - NTUA\Thesis/data/v4/high_heterogeneous_v4/high_heterogeneous_df3.csv")
    train_df4 = pd.read_csv("C:/Nikolaos Sintoris/01) Education/Msc DSML  - NTUA\Thesis/data/v4/high_heterogeneous_v4/high_heterogeneous_df4.csv")
    train_df5 = pd.read_csv("C:/Nikolaos Sintoris/01) Education/Msc DSML  - NTUA\Thesis/data/v4/high_heterogeneous_v4/high_heterogeneous_df5.csv")
    test_df = pd.read_csv("C:/Nikolaos Sintoris/01) Education/Msc DSML  - NTUA/Thesis\data/v4/high_heterogeneous_v4/final_test_df.csv")

    # train_df1 = pd.read_csv("C:/Nikolaos Sintoris/01) Education/Msc DSML  - NTUA/Thesis/data/new_final/high_heterogeneous/high_heterogeneous_df1.csv")
    # train_df2 = pd.read_csv("C:/Nikolaos Sintoris/01) Education/Msc DSML  - NTUA/Thesis/data/new_final/high_heterogeneous/high_heterogeneous_df2.csv")
    # train_df3 = pd.read_csv("C:/Nikolaos Sintoris/01) Education/Msc DSML  - NTUA/Thesis/data/new_final/high_heterogeneous/high_heterogeneous_df3.csv")
    # train_df4 = pd.read_csv("C:/Nikolaos Sintoris/01) Education/Msc DSML  - NTUA/Thesis/data/new_final/high_heterogeneous/high_heterogeneous_df4.csv")
    # train_df5 = pd.read_csv("C:/Nikolaos Sintoris/01) Education/Msc DSML  - NTUA/Thesis/data/new_final/high_heterogeneous/high_heterogeneous_df5.csv")
    # test_df = pd.read_csv("C:/Nikolaos Sintoris/01) Education/Msc DSML  - NTUA/Thesis/data/new_final/high_heterogeneous/stats_test_df.csv")

    train_df1.drop('Shannon Entropy', axis=1, inplace=True)
    train_df2.drop('Shannon Entropy', axis=1, inplace=True)
    train_df3.drop('Shannon Entropy', axis=1, inplace=True)
    train_df4.drop('Shannon Entropy', axis=1, inplace=True)
    train_df5.drop('Shannon Entropy', axis=1, inplace=True)
    test_df.drop('Shannon Entropy', axis=1, inplace=True)

    train_df1.drop('Special Characters Freq', axis=1, inplace=True)
    train_df2.drop('Special Characters Freq', axis=1, inplace=True)
    train_df3.drop('Special Characters Freq', axis=1, inplace=True)
    train_df4.drop('Special Characters Freq', axis=1, inplace=True)
    train_df5.drop('Special Characters Freq', axis=1, inplace=True)
    test_df.drop('Special Characters Freq', axis=1, inplace=True)

    # Create datasets
    X_train_1, y_train_1, X_train_2, y_train_2, X_train_3, y_train_3, X_train_4, y_train_4, X_train_5, y_train_5, X_test, y_test = create_train_test_sets(train_df1, train_df2, train_df3, train_df4, train_df5, test_df)

    # Preprocess dataset.
    X_train_1, y_train_1, X_validation_1, y_validation_1, X_test_1, y_test_1 = MinMax_normalization(X_train_1,y_train_1, X_test,y_test)
    X_train_2, y_train_2, X_validation_2, y_validation_2, X_test_2, y_test_2 = MinMax_normalization(X_train_2,y_train_2, X_test,y_test)
    X_train_3, y_train_3, X_validation_3, y_validation_3, X_test_3, y_test_3 = MinMax_normalization(X_train_3,y_train_3, X_test,y_test)
    X_train_4, y_train_4, X_validation_4, y_validation_4, X_test_4, y_test_4 = MinMax_normalization(X_train_4,y_train_4, X_test,y_test)
    X_train_5, y_train_5, X_validation_5, y_validation_5, X_test_5, y_test_5 = MinMax_normalization(X_train_5,y_train_5, X_test,y_test)

    n_features = X_train_1.shape[1]

    #
    # Train federated model.
    # # Create global model.
    # global_model = create_model(n_features)
    #
    # # Get the weights of the global model
    # global_model_weights = global_model.get_weights()
    #
    # model_1 = create_model(n_features)
    # model_2 = create_model(n_features)
    # model_3 = create_model(n_features)
    # model_4 = create_model(n_features)
    # model_5 = create_model(n_features)
    #
    # # Init the weights of the models with the weights of the global model
    # model_1.set_weights(global_model_weights)
    # model_2.set_weights(global_model_weights)
    # model_3.set_weights(global_model_weights)
    # model_4.set_weights(global_model_weights)
    # model_5.set_weights(global_model_weights)
    #
    # no_clients = 5
    #
    # train_loss_1_lst = []
    # validation_loss_1_lst = []
    # train_loss_2_lst = []
    # validation_loss_2_lst = []
    # train_loss_3_lst = []
    # validation_loss_3_lst = []
    # train_loss_4_lst = []
    # validation_loss_4_lst = []
    # train_loss_5_lst = []
    # validation_loss_5_lst = []
    #
    # for current_round in range(0, 100):
    #
    #     print('Round: {}'.format(current_round))
    #
    #     model_1, model_1_weights, train_loss_1, validation_loss_1 = train_model(model_1, X_train_1, y_train_1,
    #                                                                             X_validation_1, y_validation_1)
    #     train_loss_1_lst.append(train_loss_1)
    #     validation_loss_1_lst.append(validation_loss_1)
    #
    #     model_2, model_2_weights, train_loss_2, validation_loss_2 = train_model(model_2, X_train_2, y_train_2,
    #                                                                             X_validation_2, y_validation_2)
    #     train_loss_2_lst.append(train_loss_2)
    #     validation_loss_2_lst.append(validation_loss_2)
    #
    #     model_3, model_3_weights, train_loss_3, validation_loss_3 = train_model(model_3, X_train_3, y_train_3,
    #                                                                             X_validation_3, y_validation_3)
    #     train_loss_3_lst.append(train_loss_3)
    #     validation_loss_3_lst.append(validation_loss_3)
    #
    #     model_4, model_4_weights, train_loss_4, validation_loss_4 = train_model(model_4, X_train_4, y_train_4,
    #                                                                             X_validation_4, y_validation_4)
    #     train_loss_4_lst.append(train_loss_4)
    #     validation_loss_4_lst.append(validation_loss_4)
    #
    #     model_5, model_5_weights, train_loss_5, validation_loss_5 = train_model(model_5, X_train_5, y_train_5,
    #                                                                             X_validation_5, y_validation_5)
    #     train_loss_5_lst.append(train_loss_5)
    #     validation_loss_5_lst.append(validation_loss_5)
    #
    #     # Compute new weights.
    #     sum_of_weights = []
    #     for i in range(len(model_3_weights)):
    #         sum_of_weights.append(
    #             model_1_weights[i] + model_2_weights[i] + model_3_weights[i] + model_4_weights[i] + model_5_weights[i])
    #
    #     final_weights = []
    #     for i in range(len(sum_of_weights)):
    #         final_weights.append(sum_of_weights[i] / no_clients)
    #
    #     # Update weights
    #     global_model.set_weights(final_weights)
    #     model_1.set_weights(final_weights)
    #     model_2.set_weights(final_weights)
    #     model_3.set_weights(final_weights)
    #     model_4.set_weights(final_weights)
    #     model_5.set_weights(final_weights)
    #
    # # Evaluate global model to the test sets.
    # evaluate_model(global_model, X_test_1, y_test_1)
    #
    # evaluate_model(global_model, X_test_2, y_test_2)
    #
    # evaluate_model(global_model, X_test_3, y_test_3)
    #
    # evaluate_model(global_model, X_test_4, y_test_4)
    #
    # evaluate_model(global_model, X_test_5, y_test_5)


    # Fix path.
    models_save_path = "C:/Nikolaos Sintoris/01) Education/Msc DSML  - NTUA/Thesis/models/high_heterogeneous_V3"

    train_client_model("client_1", n_features, X_train_1, y_train_1, X_validation_1, y_validation_1, X_test_1, y_test_1,
                       models_save_path)

    train_client_model("client_2", n_features, X_train_2, y_train_2, X_validation_2, y_validation_2, X_test_2, y_test_2,
                       models_save_path)


    train_client_model("client_3", n_features, X_train_3, y_train_3, X_validation_3, y_validation_3, X_test_3, y_test_3,
                       models_save_path)

    train_client_model("client_4", n_features, X_train_4, y_train_4, X_validation_4, y_validation_4, X_test_4, y_test_4,
                       models_save_path)

    train_client_model("client_5", n_features, X_train_5, y_train_5, X_validation_5, y_validation_5, X_test_5, y_test_5,
                       models_save_path)
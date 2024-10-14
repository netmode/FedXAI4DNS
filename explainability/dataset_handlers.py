''' Standard Libraries '''
import os

''' Other Python Libraries '''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

''' Local Libraries '''
from helpers import debug


def load_dataset(filename):
    ''' Load the dataset as csv 
        Returns a dataframe the names of the features
    '''

    df = pd.read_csv(filename)
    column_headers = (
            pd.read_csv(filename, index_col = False, nrows = 0)
            .columns
            .tolist()
            )
    feature_names = column_headers[0:-3]

    debug('The dataset has been loaded!')

    return df, feature_names


def drop_correlated_features(df, correlation_threshold):
    ''' Calculate correlation coefficients for all feature pairs,
          then randomly remove one feature from the correlated pair
        Returns the dropped feature names, 
          the new dataframe and the remaining feature names
    '''

    debug('Correlation threshold is:', correlation_threshold)

    # Calculate correlation coefficients for pairs of features
    df_for_corr = df.drop(labels = ['Name', 'Label', 'Family'], axis = 1)
    correlation_coeffs = df_for_corr.corr()

    # Keep the upper triangular matrix of correlation coefficients
    upper_tri = (
            correlation_coeffs
            .where(
                np.triu(np.ones(correlation_coeffs.shape), k = 1)
                .astype(np.bool)
                )
            )

    # Drop features with high correlation (randomly one feature from the pair)
    dropped_features = [
            column for column in upper_tri.columns \
                    if any(abs(upper_tri[column]) >= correlation_threshold)
                    ]

    df_updated = df.drop(columns = dropped_features, inplace = False)
    features_remaining = df_updated.columns.tolist()[0:-3]

    debug('Feature correlation performed and excessive features removed!')
    debug('Dropped features are:', dropped_features)

    return dropped_features, df_updated, features_remaining


def split_dataset(df, test_ratio):
    ''' Split the dataset into training and test sets 
        Splits them in the X and y portions
        Returns the X and y portions of the train and test sets
    '''

    train_set, test_set = train_test_split(
            df, test_size = test_ratio, random_state = 2345, shuffle = True
            )

    X_train = train_set.iloc[:, :-3]
    y_train = train_set.iloc[:, -2:-1]
    X_test = test_set.iloc[:, :-3]
    y_test = test_set.iloc[:, -3:]

    return X_train, y_train, X_test, y_test


def scale_dataset(X_train, X_test):
    ''' Scale the dataset using min-max scaling 
        Returns the scaled X portions of the training and test sets
    '''

    minimum = X_train.min()
    maximum = X_train.max()
    X_train = (X_train - minimum) / (maximum - minimum)
    X_test = (X_test - minimum) / (maximum - minimum)

    return X_train, X_test, minimum, maximum


def get_dga_sampling_points(X_test, y_test):
    ''' Gather testing sampling points from all DGA families '''

    test_merged = pd.merge(
            left = X_test, left_index = True,\
                    right = y_test, right_index = True, how = 'inner'
            )

    sampling_points = test_merged[
            test_merged.iloc[:, -1] != 'tranco'
            ].iloc[:, :-2]

    return sampling_points


def get_specific_sampling_points(X_test, y_test, family):
    ''' Gather testing sampling points originating from specific families '''

    test_merged = pd.merge(
            left = X_test, left_index = True,\
                    right = y_test, right_index = True, how = 'inner'
            )

    sampling_points = test_merged[test_merged.iloc[:, -1] == str(family)]
    sampling_points = sampling_points.iloc[:, :-2]

    return sampling_points

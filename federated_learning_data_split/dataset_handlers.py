''' Standard Libraries '''
import os

''' Other Python Libraries '''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

''' Local Libraries '''
from helpers import debug, delete_file

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
        Returns the training and testing set
    '''

    train_set, test_set = train_test_split(
            df, test_size = test_ratio, random_state = 2345, shuffle = True
            )

    return train_set, test_set

def export_test_set(test_set):
    ''' Export testing set in Pickle format 
        Returns nothing
    '''

    delete_file('test_set.pkl')
    test_set.to_pickle('./test_set.pkl')
    debug('Done exporting the test set in Pickle format')

    return None

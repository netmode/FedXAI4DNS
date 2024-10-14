#!/usr/bin/env python3.9

''' Python Standard Libraries '''
import os
import sys

''' Local Libraries '''
from env import (
        DATASET_FILENAME, CORRELATION_THRESHOLD, TEST_RATIO,
        SHARED_FAMILY_RATIOS
        )
from helpers import debug, del_move_to_directory
from splitters import (
        split_family, split_dgas_randomly, 
        split_dgas_dedicated, split_dgas_semirandomly
        )
from dataset_handlers import(
        load_dataset, drop_correlated_features, split_dataset, export_test_set
        )
from df_helpers import export_dfs

if __name__ == '__main__':
    df, feature_names = load_dataset(DATASET_FILENAME)
    debug('Before correlation -> The dataframe is:', df)
    debug('Before correlation: The shape of the dataframe is:', df.shape)
    debug('Before correlation: The names of the features are:', feature_names)

    dropped_features, df, features_remaining = \
            drop_correlated_features(df, CORRELATION_THRESHOLD)
    debug('Dropped features because of correlation:', str(dropped_features))
    debug('After correlation: The new dataframe is:', df)
    debug('After correlation: The shape of the dataframe is:', df.shape)
    debug('After correlation: The names of the features are:', features_remaining)

    train_set, test_set = split_dataset(df, TEST_RATIO)
    debug('The size of the training set is:', train_set.shape)
    debug('The size of the testing set is:', test_set.shape)

    export_test_set(test_set)

    debug('Splitting Tranco names across participants')
    participants_tranco_listdf = split_family(train_set, 'tranco')

    debug('Removing Tranco from the current dataframe')
    df_notranco = train_set.loc[
            train_set['Family'] != 'tranco'
            ]
    debug('The new dataframe is:', df_notranco)

    debug('Splitting DGA families randomly across participants')
    del_move_to_directory('random_dga_split')
    participants_dgas_listdf = split_dgas_randomly(df_notranco)
    export_dfs(participants_tranco_listdf, participants_dgas_listdf)
    os.chdir('..')  # Return to main.py directory
    debug('Done with randomly splitting data to participants')

    debug('Splitting DGA families across participants by dedicating families')
    del_move_to_directory('dedicated_dga_split')
    participants_dgas_listdf = split_dgas_dedicated(df_notranco)
    export_dfs(participants_tranco_listdf, participants_dgas_listdf)
    os.chdir('..')
    debug('Done with dedicated splitting of data to participants')

    debug('Spliting DGA with some random and some families dedicated')
    for ratio in SHARED_FAMILY_RATIOS:
        debug('Working on case for ratio:', ratio)
        directory_name = 'semirandom_dga_split_' + str(ratio).replace('.', '')
        del_move_to_directory(directory_name)
        participants_dgas_listdf = split_dgas_semirandomly(df_notranco, ratio)
        export_dfs(participants_tranco_listdf, participants_dgas_listdf)
        os.chdir('..')
        debug('Done with semirandom splitting of data - Ratio:', ratio)
    debug('Done with semirandom splitting - all ratios')

    debug('ALL DONE!!!')

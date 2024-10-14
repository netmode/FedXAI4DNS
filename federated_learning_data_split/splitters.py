''' Standard Python Libraries '''
from random import randint
from math import ceil

''' Non-standard Python Libraries '''
import pandas as pd
import numpy as np

''' Local Libraries '''
from env import PARTICIPANTS, CONSIDERED_FAMILIES
from helpers import debug
from df_helpers import concat_dfs

def split_family(df, family_name):
    ''' Keeps a specific family from the dataframe
        and splits it across participants
        Returns a list of dataframes
    '''

    debug('Splitting names of family:', family_name)

    selected_rows = df.loc[
            df['Family'] == family_name
            ]

    rows_shuffled_df = selected_rows.sample(frac = 1)
    participant_dfs = np.array_split(rows_shuffled_df, PARTICIPANTS)

    debug('Done splitting names of family:', family_name)

    return participant_dfs

def split_dgas_randomly(df):
    ''' Splits the given DGA names randomly across participants
        DGA families repeated among participants
        Returns list of dataframes  with DGAs for each participant
    '''

    rows_shuffled = df.sample(frac = 1)
    participant_dfs = np.array_split(rows_shuffled, PARTICIPANTS)

    return participant_dfs

def split_dgas_dedicated(df):
    ''' Splits the given DGA names across participants after the names
        of whole DGA families are dedicated to a specific participant
        Returns list of dataframes with DGAs for each participant
    '''

    participant_dfs = []
    # Empty dataframes for each participant
    for index in range(PARTICIPANTS):
        pd_temp = pd.DataFrame()
        participant_dfs.append(pd_temp)

    for family_name in CONSIDERED_FAMILIES:
        selected_participant = randint(0, PARTICIPANTS - 1)
        selected_rows = df.loc[
                df['Family'] == family_name
                ]
        participant_dfs[selected_participant] = concat_dfs(
                participant_dfs[selected_participant], selected_rows
                )

    return participant_dfs

def split_dgas_semirandomly(df, ratio):
    ''' Splits the given DGA names across participants after combining
        the random method and the dedicated method. The common family
        ratio specifies how many families will be shared among participants
        and thus, their names will be randomly split across participants
        Returns list of dataframes with DGAs for each participant
    '''

    total_families_number = len(CONSIDERED_FAMILIES)
    common_families_number = ceil(ratio * total_families_number)
    shared_families = CONSIDERED_FAMILIES[0:common_families_number]
    non_shared_families = CONSIDERED_FAMILIES[common_families_number:]
    debug('Number of common families:', common_families_number)
    debug('Number of disjoint families:', len(non_shared_families))
    debug('Total number of families:', total_families_number)
    debug('The following families will be common:', shared_families)
    debug('The following families will not be common:', non_shared_families)

    debug('Applying Pandas masking to DataFrames')
    mask = df['Family'].isin(shared_families)
    df_shared_families = df[mask]
    debug('DataFrame with shared families:', df_shared_families)
    
    mask = df['Family'].isin(non_shared_families)
    df_non_shared_families = df[mask]
    debug('DataFrame with non shared families:', df_non_shared_families)

    participant_dfs_random = split_dgas_randomly(df_shared_families)
    debug('DataFrame from the random process:', participant_dfs_random)
    participant_dfs_dedicated = split_dgas_dedicated(df_non_shared_families)
    debug('DataFrame from the dedicated process:', participant_dfs_dedicated)

    # Concatenate the two DGA DataFrames for each participant
    participant_dfs = []
    for participant in range(PARTICIPANTS):
        concatenated_df = concat_dfs(
            participant_dfs_random[participant],
            participant_dfs_dedicated[participant]
            )
        participant_dfs.append(concatenated_df)

    return participant_dfs

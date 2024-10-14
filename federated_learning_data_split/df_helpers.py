''' Python Non-standard Libraries '''
import pandas as pd

''' Local Libraries '''
from env import PARTICIPANTS

def concat_dfs(df1, df2):
    ''' Concats two dataframes
        Returns the concatenated dataframe
    '''

    concatenated_df = pd.concat([df1, df2])

    return concatenated_df

def export_dfs(df1_list, df2_list):
    ''' Concatenates the dataframes and exports them to csv's '''

    for index in range(PARTICIPANTS):
        df_concat = concat_dfs(
                df1_list[index],
                df2_list[index],
                )

        filename = 'participant' + str(index) + '.csv'
        df_concat.to_csv(filename, header = True, index = False)

    return None

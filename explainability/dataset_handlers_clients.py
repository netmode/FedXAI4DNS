''' Python Non-Standard Libraries '''
import pandas as pd

''' Local Libraries '''
from helpers import debug
from env import PARTICIPANTS


def load_client_datasets(file_path):
    ''' Loads the datasets of the Federated Learning clients
        file_path defines the path containing the datasets
    '''

    dfX_list = []  # features only
    dfy_list = []  # label only, i.e. 0 or 1 for binary classification
    dfy_extended_list = [] # labels plus DGA family and domain name

    path = file_path + 'participant{}.csv'

    for participant in range(PARTICIPANTS):
        filename = path.format(participant)
        df = pd.read_csv(filename)
        X = df.iloc[:, :-3]
        y = df.iloc[:, -2:-1]
        y_extended = df.iloc[:, -3:]
        dfX_list.append(X)
        dfy_list.append(y)
        dfy_extended_list.append(y_extended)

    return dfX_list, dfy_list, dfy_extended_list


def scale_client_datasets(dfX_list, minimum, maximum):
    ''' Scales the dataset of a Federated Learning client
        minimum and maximum come from the non federated model
    '''
    
    dfX_list_updated = []
    for participant in range(PARTICIPANTS):
        df_temp = dfX_list[participant]
        df_temp = (df_temp - minimum) / (maximum - minimum)
        dfX_list_updated.append(df_temp)

    return dfX_list_updated


def process_client_datasets(path_suffix, minimum, maximum):
    ''' Complete path for loading and preprocessing Federated Learning
        client datasets
    '''

    path = '../federated_learning_data_split/' + path_suffix
    dfX_list, dfy_list, dfy_extended_list = load_client_datasets(path)
    dfX_list = scale_client_datasets(dfX_list, minimum, maximum)

    return dfX_list, dfy_list, dfy_extended_list

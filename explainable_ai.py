import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import load_model

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


def create_shap_test_dataset(test_df):
    '''
        Create test dataset to use it in XAI.

            Input:
                test_df: A dataframe with the test set

            Return:
                X_test_shap_np: A numpy array with 200 random malicious data samples from test dataset
    '''
    # Keep only malicious data samples.
    malicious_test_df = test_df[test_df['Label'] == 1]

    # Keep only 200 random data samples
    new_test_df = malicious_test_df.sample(n=200, random_state=42)
    print(new_test_df)
    print(new_test_df['Family'].value_counts())
    # print(new_test_df.groupby('Family')['Vowels Freq'].mean().round().reset_index())
    # print("Benign")
    # benign_test_df = test_df[test_df['Label'] == 0]
    # # Keep only 200 random data samples
    # new_benign_test_df = benign_test_df.sample(n=200, random_state=42)
    # print(new_benign_test_df.groupby('Family')['Vowels Freq'].mean().round().reset_index())

    # Drop domain name, Family and Label column to create X_test data.
    X_test_shap_np = np.array(new_test_df.drop(["Domain Name", "Family", "Label"], axis=1))

    return X_test_shap_np

def compute_shap_values(model, X_train, X_test_shap):
    '''
        Compute shap values for a client.

            Input:
                model: Client's model
                X_train: Clients training dataset
                X_test_shap: Test dataset for xai

            Return:
                shap_values: A numpy array with the shap values
    '''
    # Create background train dataset
    kmeans_obj = KMeans(n_clusters=50, init='k-means++', random_state=0, n_init="auto").fit(X_train)
    background_train_dataset = kmeans_obj.cluster_centers_

    # Use the Kernel SHAP explainer
    explainer_obj = shap.KernelExplainer(model.predict, background_train_dataset)

    # Get SHAP values
    shap_values = explainer_obj.shap_values(X_test_shap)

    return shap_values[0]

def explainable_ai_clients(model, X_train, X_test_shap_np, feature_names_lst, shap_images_save_path, client_name):
    '''
        Explainable AI pipeline for a client.

            Input:
                model: Client's model
                X_train: Clients training dataset
                X_test_shap_np: Test dataset for xai
                feature_names_lst: A list with all the feature names
                shap_images_save_path: The directory to save the images
                client_name: Name of the current client

            Return:
                Print and save summary plot
    '''
    # Compute shap values
    shap_values = compute_shap_values(model, X_train, X_test_shap_np)

    # Summary_plot
    shap.summary_plot(shap_values, X_test_shap_np, plot_type="dot", feature_names=feature_names_lst)
    # plt.savefig(shap_images_save_path + '/' + client_name + '.png')
    plt.show()

    return shap_values

if __name__ == '__main__':

    # Load Dataset.
    train_df1 = pd.read_csv("C:/Nikolaos Sintoris/01) Education/Msc DSML  - NTUA\Thesis/data/v4/high_heterogeneous_v4/high_heterogeneous_df1.csv")
    train_df2 = pd.read_csv("C:/Nikolaos Sintoris/01) Education/Msc DSML  - NTUA\Thesis/data/v4/high_heterogeneous_v4/high_heterogeneous_df2.csv")
    train_df3 = pd.read_csv("C:/Nikolaos Sintoris/01) Education/Msc DSML  - NTUA\Thesis/data/v4/high_heterogeneous_v4/high_heterogeneous_df3.csv")
    train_df4 = pd.read_csv("C:/Nikolaos Sintoris/01) Education/Msc DSML  - NTUA\Thesis/data/v4/high_heterogeneous_v4/high_heterogeneous_df4.csv")
    train_df5 = pd.read_csv("C:/Nikolaos Sintoris/01) Education/Msc DSML  - NTUA\Thesis/data/v4/high_heterogeneous_v4/high_heterogeneous_df5.csv")
    test_df = pd.read_csv("C:/Nikolaos Sintoris/01) Education/Msc DSML  - NTUA/Thesis\data/v4/high_heterogeneous_v4/final_test_df.csv")

    # train_df1['Shannon Entropy'] = -train_df1['Shannon Entropy']
    # train_df2['Shannon Entropy'] = -train_df2['Shannon Entropy']
    # train_df3['Shannon Entropy'] = -train_df3['Shannon Entropy']
    # train_df4['Shannon Entropy'] = -train_df4['Shannon Entropy']
    # train_df5['Shannon Entropy'] = -train_df5['Shannon Entropy']
    # test_df['Shannon Entropy'] = -test_df['Shannon Entropy']

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
    X_train_1, y_train_1, X_train_2, y_train_2, X_train_3, y_train_3, X_train_4, y_train_4, X_train_5, y_train_5, X_test, y_test = create_train_test_sets(
        train_df1, train_df2, train_df3, train_df4, train_df5, test_df)

    # Preprocess dataset.
    X_train_1, y_train_1, X_validation_1, y_validation_1, X_test_1, y_test_1 = MinMax_normalization(X_train_1,y_train_1, X_test,y_test)
    X_train_2, y_train_2, X_validation_2, y_validation_2, X_test_2, y_test_2 = MinMax_normalization(X_train_2,y_train_2, X_test,y_test)
    X_train_3, y_train_3, X_validation_3, y_validation_3, X_test_3, y_test_3 = MinMax_normalization(X_train_3,y_train_3, X_test,y_test)
    X_train_4, y_train_4, X_validation_4, y_validation_4, X_test_4, y_test_4 = MinMax_normalization(X_train_4,y_train_4, X_test,y_test)
    X_train_5, y_train_5, X_validation_5, y_validation_5, X_test_5, y_test_5 = MinMax_normalization(X_train_5,y_train_5, X_test,y_test)

    # Load pre-trained models.
    n_features = X_train_1.shape[1]

    # Fix path.
    models_save_path = "C:/Nikolaos Sintoris/01) Education/Msc DSML  - NTUA/Thesis/models/high_heterogeneous_V3"

    model_1 = load_model(models_save_path + '/' + 'client_1.h5')
    model_2 = load_model(models_save_path + '/' + 'client_2.h5')
    model_3 = load_model(models_save_path + '/' + 'client_3.h5')
    model_4 = load_model(models_save_path + '/' + 'client_4.h5')
    model_5 = load_model(models_save_path + '/' + 'client_5.h5')

    feature_names_lst = train_df1.drop(['Domain Name', 'Family', 'Label'], axis=1).columns

    # Create xai test dataset.
    X_test_shap_np = create_shap_test_dataset(test_df)

    # Fix path.
    shap_images_save_path = "C:/Nikolaos Sintoris/01) Education/Msc DSML  - NTUA/Thesis/results/shap_results_no_shannon_special"

    clients_shap_values_lst = []

    clients_shap_values_lst.append(explainable_ai_clients(model_1, X_train_1, X_test_shap_np, feature_names_lst, shap_images_save_path,
                           'client_1_summary_plot'))
    clients_shap_values_lst.append(explainable_ai_clients(model_2, X_train_2, X_test_shap_np, feature_names_lst, shap_images_save_path,
                           'client_2_summary_plot'))
    clients_shap_values_lst.append(explainable_ai_clients(model_3, X_train_3, X_test_shap_np, feature_names_lst, shap_images_save_path,
                           'client_3_summary_plot'))
    clients_shap_values_lst.append(explainable_ai_clients(model_4, X_train_4, X_test_shap_np, feature_names_lst, shap_images_save_path,
                           'client_4_summary_plot'))
    clients_shap_values_lst.append(explainable_ai_clients(model_5, X_train_5, X_test_shap_np, feature_names_lst, shap_images_save_path,
                           'client_5_summary_plot'))

    # Calculate the average shap values
    federated_model_shap_values_np = np.mean(np.array(clients_shap_values_lst), axis=0)

    # summary_plot of a specific class
    shap.summary_plot(federated_model_shap_values_np, X_test_shap_np, plot_type="dot", feature_names=feature_names_lst)
    plt.show()
''' Python Non-Standard Libraries '''
import shap
import numpy as np

''' Python Local Libraries '''
from helpers import debug
from env import (
        EXPLAINABILITY_BACKGROUND_INSTANCES, EXPLAINABILITY_TEST_INSTANCES,
        PARTICIPANTS
        )


def get_background_set(X_train):
    ''' Extracts the eXplainability Background Instances (XBIs) by applying 
        the K-Means clustering algorithm on the training set instances
    '''

    debug('Extracting XBIs')
    background_set = shap.kmeans(X_train, EXPLAINABILITY_BACKGROUND_INSTANCES)

    return background_set


def get_background_set_federated(X_train_list):
    ''' Similar to the previous, but for multiple clients '''

    debug('Extracting XBIs for the clients')

    xbis_clients = [None] * PARTICIPANTS

    for participant in range(PARTICIPANTS):
        background_set = shap.kmeans(
                X_train_list[participant], EXPLAINABILITY_BACKGROUND_INSTANCES
                )
        xbis_clients[participant] = background_set

    return xbis_clients


def derive_model_explainer(model, background_set):
    ''' Applies the SHAP Kernel Explainer on the model and the xbis '''

    debug('Deriving model explainer')
    model_explainer = shap.KernelExplainer(model.predict, background_set)

    return model_explainer


def derive_model_explainer_federated(models, background_sets):
    ''' Similar to the previous, but for multiple cases '''

    debug('Deriving model explainers for federated clients')
    model_explainers = [None] * PARTICIPANTS

    for participant in range(PARTICIPANTS):
        model_explainer =\
                shap.KernelExplainer(
                        models[participant].predict,
                        background_sets[participant]
                        )
        model_explainers[participant] = model_explainer

    return model_explainers


def derive_xtis_via_sampling(testing_sampling_points):
    ''' Derive XTIs by randomly sampling a subset of the testing examples '''

    set_size = len(testing_sampling_points)
    if set_size < EXPLAINABILITY_TEST_INSTANCES:
        xti_number = set_size
    else:
        xti_number = EXPLAINABILITY_TEST_INSTANCES

    xtis = shap.utils.sample(
            testing_sampling_points, xti_number, random_state = 1452
            )

    # Separate features from names
    xti_names = xtis.iloc[:, -1]
    xti_features = xtis.iloc[:, 0:-1]

    return xti_names, xti_features


def extract_shap_values(model_explainer, xti_features):
    ''' Extracts SHAP values based on the model explainer and the XTIs '''

    model_shap_values = model_explainer.shap_values(xti_features)
    model_shap_values = np.asarray(model_shap_values)
    model_shap_values = model_shap_values[0]

    return model_shap_values

def extract_shap_values_federated(model_explainers, xti_features):
    ''' Extracts SHAP values for the Federated Learning clients '''

    client_shap_values = [None] * PARTICIPANTS

    for participant in range(PARTICIPANTS):
        client_shap_values[participant] =\
                model_explainers[participant].shap_values(xti_features)
        client_shap_values[participant] =\
                np.asarray(client_shap_values[participant])
        client_shap_values[participant] = client_shap_values[participant][0]

    return client_shap_values

def aggregate_shap_values(client_shap_values):
    ''' Aggregated the SHAP values of the Federated Learning clients '''

    for participant in range(PARTICIPANTS):
        debug('Client SHAP values:', client_shap_values[participant])

    summation = np.array(client_shap_values[0])
    for participant in range(1, PARTICIPANTS):
        summation += np.array(client_shap_values[participant])

    average = summation / PARTICIPANTS
    debug('Global SHAP values:', average)

    return average

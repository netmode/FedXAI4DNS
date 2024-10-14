from env import (
        DATASET_FILENAME, CORRELATION_THRESHOLD, TEST_RATIO, PARTICIPANTS,
        SELECTED_FAMILIES
        )
from helpers import debug, delete_directory, create_directory
from trainers import (
        non_federated_trainer, federated_trainer, nonfed_local_model_trainer
        )
from evaluators import (
        evaluate_nonfed_model, evaluate_fed_model,
        evaluate_nonfed_model_clients
        )
from plots import summary_plots, dependence_plots, force_plots
from shap_handler import (
        get_background_set, derive_model_explainer, derive_xtis_via_sampling,
        extract_shap_values, get_background_set_federated, 
        derive_model_explainer_federated, extract_shap_values_federated,
        aggregate_shap_values
        )
from dataset_handlers import (
        load_dataset, drop_correlated_features, split_dataset, scale_dataset,
        get_dga_sampling_points, get_specific_sampling_points
        )
from dataset_handlers_clients import process_client_datasets


def pipeline(dirname, accfile, prefix, minim, maxim, xti_f, X_test, y_test):
    ''' The pipeline for the Federated Learning cases '''

    delete_directory(prefix)
    create_directory(prefix)

    dfX_list, dfy_list, dfy_extended_list =\
            process_client_datasets(dirname, minim, maxim)
    local_models, global_model =\
            federated_trainer(dfX_list, dfy_list)
    evaluate_fed_model(
            local_models, global_model, X_test, y_test, accfile
            )
    xbis_clients = get_background_set_federated(dfX_list)
    model_explainers_fed =\
            derive_model_explainer_federated (local_models, xbis_clients)
    client_shap_values =\
            extract_shap_values_federated(model_explainers_fed, xti_f)
    global_shap_values = aggregate_shap_values(client_shap_values)

    summary_plots(
            global_shap_values,
            'all-dgas',
            xti_features,
            prefix,
            'global-'
            )

    dependence_plots(
            global_shap_values,
            'all-dgas',
            xti_features,
            prefix,
            'global-'
            )

    for participant in range(PARTICIPANTS):
        client_path = 'local-' + str(participant) + '-'
        summary_plots(
                client_shap_values[participant],
                'all-dgas',
                xti_features,
                prefix,
                client_path
                )

        dependence_plots(
                client_shap_values[participant],
                'all-dgas',
                xti_features,
                prefix,
                client_path
                )

    # Train local models in a non-federated manner
    local_models_nonfed = nonfed_local_model_trainer(dfX_list, dfy_list)
    filename = 'local' + accfile
    evaluate_nonfed_model_clients(
            local_models_nonfed, X_test, y_test, filename
            )
    model_explainers_nonfed =\
            derive_model_explainer_federated (
                    local_models_nonfed, xbis_clients
                    )
    client_shap_values_nonfed =\
            extract_shap_values_federated(model_explainers_nonfed, xti_f)
                
    for participant in range(PARTICIPANTS):
        client_path = 'nonfed-local-' + str(participant) + '-'
        summary_plots(
                client_shap_values_nonfed[participant],
                'all-dgas',
                xti_features,
                prefix,
                client_path
                )

        dependence_plots(
                client_shap_values_nonfed[participant],
                'all-dgas',
                xti_features,
                prefix,
                client_path
                )

    return None


if __name__ == '__main__':
    # Delete the results folder
    delete_directory('xai_results')
    create_directory('xai_results')
    delete_directory('accuracy_results')
    create_directory('accuracy_results')

    # Operations on the dataset
    df, feature_names = load_dataset(DATASET_FILENAME)
    dropped_features, df, features_remaining =\
            drop_correlated_features(df, CORRELATION_THRESHOLD)
    X_train, y_train, X_test, y_test = split_dataset(df, TEST_RATIO)
    X_train, X_test, minimum, maximum = scale_dataset(X_train, X_test)

    # Train model without Federated Learning and evaluate it
    model = non_federated_trainer(X_train, y_train)
    evaluate_nonfed_model(model, X_test, y_test)
    background_xbis_nonfed = get_background_set(X_train)
    model_explainer_nonfed =\
            derive_model_explainer(model, background_xbis_nonfed)

    # Choose testing sampling points to explain for the non-federated model
    all_dgas_sampling_points = get_dga_sampling_points(X_test, y_test)
    xti_names, xti_features =\
            derive_xtis_via_sampling(all_dgas_sampling_points)
    shap_values = extract_shap_values(model_explainer_nonfed, xti_features)
    summary_plots(shap_values, 'all-dgas', xti_features, 'xai_results', '')
    dependence_plots(shap_values, 'all-dgas', xti_features, 'xai_results', '')
    force_plots(
            model,
            shap_values,
            'all-dgas',
            xti_features,
            xti_names,
            model_explainer_nonfed,
            'xai_results',
            ''
            )

    # The same procedure for the specific DGA families
    for selected_family in SELECTED_FAMILIES:
        selected_sampling_points =\
                get_specific_sampling_points(X_test, y_test, selected_family)
        selected_xti_names, selected_xti_features =\
                derive_xtis_via_sampling(selected_sampling_points)
        selected_shap_values = extract_shap_values(
                model_explainer_nonfed, selected_xti_features
                )
        summary_plots(
                selected_shap_values,
                selected_family,
                selected_xti_features,
                'xai_results',
                ''
                )
        dependence_plots(
                selected_shap_values,
                selected_family,
                selected_xti_features,
                'xai_results',
                ''
                )

    # Train client models, i.e. via Federated Learning (case random split)
    pipeline(
            'random_dga_split/',
            'random_split.txt',
            'random_split',
            minimum,
            maximum,
            xti_features,
            X_test,
            y_test
            )

    # Train client models, i.e. via Federated Learning (case dedicated split)
    pipeline(
            'dedicated_dga_split/',
            'dedicated_split.txt',
            'dedicated_split',
            minimum,
            maximum,
            xti_features,
            X_test,
            y_test
            )

    # Train client models, i.e. via Federated Learning (semirandom 0.25)
    pipeline(
            'semirandom_dga_split_025/',
            'sr_025_split.txt',
            'sr_025_split',
            minimum,
            maximum,
            xti_features,
            X_test,
            y_test
            )

    # Train client models, i.e. via Federated Learning (semirandom 0.5)
    pipeline(
            'semirandom_dga_split_05/',
            'sr_05_split.txt',
            'sr_05_split',
            minimum,
            maximum,
            xti_features,
            X_test,
            y_test
            )

    # Train client models, i.e. via Federated Learning (semirandom 0.75)
    pipeline(
            'semirandom_dga_split_075/',
            'sr_075_split.txt',
            'sr_075_split',
            minimum,
            maximum,
            xti_features,
            X_test,
            y_test
            )

    debug('ALL DONE!!!')

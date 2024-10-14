''' Python Standard Libraries '''
import os

''' Python Non-Standard Libraries '''
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import shap

''' Python Local Libraries '''
from helpers import create_directory


def summary_plots(shap_values, family, xti_features, prefix_path, client):
    ''' Return SHAP summary plots '''

    path = prefix_path + '/summary-plots-' + str(client) + str(family)
    create_directory(path)
    os.chdir(path)

    fig = plt.clf()
    shap.summary_plot(
            shap_values, xti_features, plot_type = "bar", show = False
            )
    name = 'summarybar-original.png'
    plt.savefig(name)
    plt.close('all')

    plt.xlim(-1, 1)
    name = 'summarybar-xlim-11.png'
    plt.savefig(name)
    plt.close('all')

    fig = plt.clf()
    shap.summary_plot(shap_values, xti_features, show = False)
    name = 'summarynotbar-original.png'
    plt.savefig(name)
    plt.close('all')

    plt.xlim(-1, 1)
    name = 'summarynotbar-xlim-11.png'
    plt.savefig(name)
    plt.close('all')

    os.chdir('../..')

    return None


def dependence_plots(shap_values, family, xti_features, prefix_path, client):
    ''' Return SHAP dependence plots for multiple features '''

    path = prefix_path + '/dependence-plots-' + str(client) + str(family)
    create_directory(path)
    os.chdir(path)

    features = [
            'Length', 'Reputation', 'Words_Mean', 'Words_Freq', 'Vowel_Freq',
            'Entropy', 'DeciDig_Freq', 'Max_DeciDig_Seq', 'Max_Let_Seq'
            ]

    for feature in features:
        fig = plt.clf()
        shap.dependence_plot(
                feature, shap_values, xti_features, show = False
                )
        name = feature + '.png'
        plt.savefig(name, bbox_inches = 'tight')
        plt.close('all')

    os.chdir('../..')

    return None


def force_plots(model, shapval, fam, xti_f, xti_n, explainer, prefix, client):
    ''' Return SHAP force plots for multiple testing instances '''

    path = prefix + '/force-plots-' + str(client) + str(fam)
    create_directory(path)
    os.chdir(path)

    predictions = model.predict(xti_f)
    index_values = list(xti_f.index.values)

    sequence = 0
    for index in index_values:
        original_name = xti_n[index]
        name = original_name.replace('.', '+')
        prediction = predictions[sequence]

        fig = plt.clf()
        shap.force_plot(
                explainer.expected_value,
                shapval[sequence, :],
                xti_f.loc[index],
                matplotlib = True,
                show = False
                )

        name_of_file = str(sequence) + '-name-' + str(name) +\
                '-prediction-' + str(prediction) + '-original.png'
        plt.title(original_name, y = 1.5)
        plt.savefig(name_of_file, bbox_inches = 'tight')
        plt.close('all')

        fig = plt.clf()
        shap.force_plot(
                explainer.expected_value,
                shapval[sequence, :],
                xti_f.loc[index],
                matplotlib = True,
                show = False,
                contribution_threshold = 0.1
                )

        name_of_file = str(sequence) + '-name-' + str(name) +\
                '-prediction-' + str(prediction) + '-threshold01.png'
        plt.title(original_name, y = 1.5)
        plt.savefig(name_of_file, bbox_inches = 'tight')
        plt.close('all')

        sequence += 1
        # Plot only the first 250 or less if no more than 250 exist
        if sequence == 250:
            break

    os.chdir('../..')

    return None

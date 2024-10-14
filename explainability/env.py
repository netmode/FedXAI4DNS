''' Python Standard Libraries '''
import random

# For debugging purposes: True to enable debugging
DEBUG_ENABLED = True
DEBUG_SEPARATOR = '---------------------------------------------'

# Epochs for the Non Federated Learning Case
NON_FED_EPOCHS = 100

# Epochs for the Federated Learning Case (per round) and rounds
FED_EPOCHS = 5
FED_ROUNDS = 30

# Correlation threshold for Pearson (calculated between feature paris)
CORRELATION_THRESHOLD = 0.9

# Testing set percentage to the whole dataset
TEST_RATIO = 0.2

# Number of Federated Learning Clients
PARTICIPANTS = 10

# K-Means clusters for eXplainability Background Instances (XBIs)
EXPLAINABILITY_BACKGROUND_INSTANCES = 50

# Testing sampling points for SHAP-based interpretations
EXPLAINABILITY_TEST_INSTANCES = 250

# The considered families
CONSIDERED_FAMILIES = [
        'dyre', 'symmi', 'ramnit', 'banjori', 'pykspa', 'murofet', 'wd',
        'simda', 'gameover', 'murofetweekly', 'nymaim', 'gozi', 'matsnu',
        'bamital', 'nymaim2', 'suppobox', 'padcrypt', 'szribi', 'torpig',
        'proslikefan', 'emotet', 'conficker', 'corebot', 'chinad', 'pitou',
        'dnschanger', 'tinynuke', 'oderoor', 'qakbot', 'ranbyus', 'tinba',
        'cryptolocker', 'qadars', 'infy', 'rovnix', 'pushdo', 'vidro',
        'urlzone', 'necurs', 'monerominer', 'virut', 'sphinx', 'pandabanker',
        'locky', 'qsnatch'
        ]
random.Random(1234).shuffle(CONSIDERED_FAMILIES)

# Specific DGA families to delve into for XAI plots
SELECTED_FAMILIES = [
        'dyre', 'banjori', 'bamital', 'suppobox', 'conficker', 'corebot',
        'matsnu', 'qsnatch', 'simda', 'necurs', 'qakbot', 'gozi', 'nymaim',
        'nymaim2', 'cryptolocker', 'dnschanger', 'ramnit', 'symmi', 'pushdo'
        ]

# Labeled Dataset (after features are calculated)
DATASET_FILENAME = 'aux_data/labeled_dataset_features.csv'

import random

''' Global Constants - not to be modified by users '''
DEBUG_SEPARATOR = "-----------------------------------------------------------"

''' Global Constants - can be modified only here '''
DEBUG_ENABLED = True  # True to enable debug, otherwise False

# Number of participants in the Federated Learning scheme
PARTICIPANTS = 5

# Testing set percentage to the whole dataset
TEST_RATIO = 0.2

# The families to consider for SHAP interpretations
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

# The dataset of names and the extracted features
DATASET_FILENAME = 'data/labeled_dataset_features.csv'

# Pearson correlation threshold - evaluated for feature pairs (one dropped)
CORRELATION_THRESHOLD = 0.9

# Shared families to the total families number (3rd case of data splitting)
SHARED_FAMILY_RATIOS = [0.25, 0.5, 0.75]

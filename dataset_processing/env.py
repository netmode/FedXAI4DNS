# A separator to help discern output lines used for debugging purposes
DEBUG_SEPARATOR = '--------------------------------------------------'

# Determines if debugging messages are enabled or not (True if yes)
DEBUG_ENABLED = True

# Username and password related to the DGArchive repository
DGARCHIVE_USERNAME = 'ENTER USERNAME HERE'
DGARCHIVE_PASSWORD = 'ENTER PASSWORD HERE'

# The link where the DGArchive dataset is found
DGARCHIVE_FILENAME = '2020-06-19-dgarchive_full.tgz'
DGARCHIVE_LINK = 'https://dgarchive.caad.fkie.fraunhofer.de/datasets/'\
        + DGARCHIVE_FILENAME

# The link from which the Tranco repository is available
TRANCO_URL = 'https://tranco-list.eu/download/4Q6GX/full'

# The link containing the Mozilla public suffix list
SUFFIX_LIST_URL = 'https://publicsuffix.org/list/public_suffix_list.dat'

# Number of names to keep from large DGA families
MAX_SIZE = 15000

# Number of names to keep from Tranco
TRANCO_SIZE = 900000

# Helpful for feature extraction
CHARS = 'abcdefghijklmnopqrstuvwxyz'
CHARS_AND_DIGITS = 'abcdefghijklmnopqrstuvwxyz0123456789'
DIGITS = '0123456789'
VOWELS = 'aeiouy'
SPECIAL_CHARS = '-_.'
FEATURE_HEADERS = 'Length,Max_DeciDig_Seq,Max_Let_Seq,Freq_A,Freq_B,Freq_C,\
Freq_D,Freq_E,Freq_F,Freq_G,Freq_H,Freq_I,Freq_J,Freq_K,Freq_L,Freq_M,Freq_N,\
Freq_O,Freq_P,Freq_Q,Freq_R,Freq_S,Freq_T,Freq_U,Freq_V,Freq_W,Freq_X,Freq_Y,\
Freq_Z,Freq_0,Freq_1,Freq_2,Freq_3,Freq_4,Freq_5,Freq_6,Freq_7,Freq_8,Freq_9,\
Spec_Char_Freq,Ratio_Spec_Char,DeciDig_Freq,Ratio_DeciDig,Vowel_Freq,\
Vowel_Ratio,Max_Gap,Reputation,Words_Freq,Words_Mean,Entropy,Name,Label,\
Family'

# Intermediate Files
AUX_DIR = 'aux_data'
PUBLIC_SUFFIX_LIST = 'aux_data/public_suffixes_list.csv'
PUBLIC_SUFFIX_LIST_V2 = 'aux_data/public_suffixes_list_v2.csv'
TRANCO_WHITELIST = 'aux_data/tranco_top100k.txt'
TRANCO_FULL_LIST = 'aux_data/tranco_full_list.csv'
DGA_FILENAME = 'aux_data/dgarchive_full.csv'
DGA_NAMES_IN_TRANCO = 'aux_data/dga_names_in_tranco.txt'
TRANCO_WITHOUT_DGA_NAMES = 'aux_data/tranco_full_original.csv'
TRANCO_REMAINING_AFTER_TOP100K = 'aux_data/tranco_remaining.txt'

# Extracted Dataset Files
INTERMEDIARY_DATASET = 'aux_data/labeled_dataset.csv'
FINAL_DATASET = 'aux_data/labeled_dataset_features.csv'

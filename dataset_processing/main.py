''' Local Libraries '''
from env import AUX_DIR, INTERMEDIARY_DATASET
from suffix_processor import (
        download_suffix_list, delete_unwanted_lines, load_suffixes
        )
from helpers import delete_directory, create_directory, debug
from reduction import reduce_main
from feature_extractor import create_final_dataset
from tranco_processor import (
        download_tranco_full_list, filter_tranco, create_tranco_whitelist
        )
from dgarchive_processor import (
        download_dgarchive, untar_dataset_and_remove_p2p, list_dga_files,
        sum_dga_files, determine_families, load_dgarchive_names
        )

if __name__ == '__main__':
    # Recreate directory aux_data that will store helpful, intermediary info
    delete_directory(AUX_DIR)
    create_directory(AUX_DIR)

    # Download and process public suffix list from Mozilla
    download_suffix_list()
    delete_unwanted_lines()

    # Download and process DGArchive
    download_dgarchive()
    untar_dataset_and_remove_p2p()
    dga_files_list = list_dga_files()
    sum_dga_files(dga_files_list)
    dga_families_list = determine_families(dga_files_list)

    debug('List of families included in the dataset', dga_families_list)

    # Download Tranco list and create the whitelist
    download_tranco_full_list()
    all_dgas_set = load_dgarchive_names()
    filter_tranco(all_dgas_set)
    create_tranco_whitelist()

    # Remove public suffixes from all domain names
    fdw = open(INTERMEDIARY_DATASET, 'w')
    suffixes = load_suffixes()
    remaining_families, total_sizes =\
            reduce_main(dga_families_list, suffixes, fdw)

    debug('Families remaining in the dataset', remaining_families)
    debug('Number of names per family', total_sizes)

    # Export features and create the final dataset
    debug('Starting feature extraction')
    create_final_dataset()
    debug('Dataset construction DONE!!!')

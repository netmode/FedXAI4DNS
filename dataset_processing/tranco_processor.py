''' Python Standard Libraries '''
import os
import sys

''' Python Non-Standard Libraries '''
import requests

''' Local Libraries '''
from helpers import debug
from env import (
        TRANCO_URL, TRANCO_FULL_LIST, DGA_NAMES_IN_TRANCO,
        TRANCO_WITHOUT_DGA_NAMES, TRANCO_WHITELIST,
        TRANCO_REMAINING_AFTER_TOP100K
        )

def download_tranco_full_list():
    ''' Downloads the full Tranco repository from the given link '''

    debug('Downloading Tranco list')
    r = requests.get(TRANCO_URL, allow_redirects = True)
    open(TRANCO_FULL_LIST, 'wb').write(r.content)
    debug('Tranco list has been downloaded')

    return None

def filter_tranco(dga_names_set):
    ''' Filters downloaded Tranco names against the DGArchive repository '''

    fdw = open(DGA_NAMES_IN_TRANCO, 'w')
    fdw2 = open(TRANCO_WITHOUT_DGA_NAMES, 'w')
    tranco_filename = TRANCO_FULL_LIST

    removed_names = set()

    debug('Reading Tranco List to find DGAs within the list')

    with open(tranco_filename) as infile:
        for line in infile:
            line2 = line.strip()
            no, name = line2.split(',')
            if name in dga_names_set:
                removed_names.add(name)
                fdw.write(name + '\n')
            else:
                fdw2.write(line)

    fdw.close()
    fdw2.close()

    debug('DGA names withing Tranco found and Tranco List cleared')

    return None
    
def create_tranco_whitelist():
    ''' Creates the whitelist for the benign N-grams '''

    fdw = open(TRANCO_WHITELIST, 'w')
    fdw2 = open(TRANCO_REMAINING_AFTER_TOP100K, 'w')
    debug("Starting to split Tranco into top 100k names and remaining ones")

    counter = 0
    with open(TRANCO_WITHOUT_DGA_NAMES) as infile:
        for line in infile:
            counter += 1
            line = line.strip()
            no, name = line.split(',')
            if counter <= 100000:
                fdw.write(name + '\n')
            else:
                fdw2.write(name + '\n')

    fdw.close()
    fdw2.close()

    debug("Done separating Tranco to top 100k and remaining names")

    return None

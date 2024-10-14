''' Python Non-Standard Libraries '''
import requests

''' Local Libraries '''
from env import SUFFIX_LIST_URL, PUBLIC_SUFFIX_LIST, PUBLIC_SUFFIX_LIST_V2
from helpers import debug

def download_suffix_list():
    ''' Downloads the list of public suffixes available from Mozilla
    '''

    r = requests.get(SUFFIX_LIST_URL, allow_redirects = True)
    open(PUBLIC_SUFFIX_LIST, 'wb'). write(r.content)

    debug('Public suffix list has been downloaded')

    return None

def delete_unwanted_lines():
    ''' Will delete some unwanted lines, e.g. lines that are empty
        or lines that start with slashes
    '''

    fdw = open(PUBLIC_SUFFIX_LIST_V2, 'w')

    with open(PUBLIC_SUFFIX_LIST, 'r') as file:
        for line in file:
            if line != "\n" and line[0] != '/':
                fdw.write(line)
    fdw.close()

    debug('Extra lines have been deleted from the public suffix list')

    return None

def load_suffixes():
    ''' Loads the public suffixes available from Mozilla '''

    debug('Loading the Mozilla Firefox suffixes in a set') 

    suffix_list = PUBLIC_SUFFIX_LIST_V2
    fdr = open(suffix_list, 'r')
    suffixes = set()

    for line in fdr:
        suffix = line.strip()
        suffixes.add(suffix)

    fdr.close()

    return suffixes


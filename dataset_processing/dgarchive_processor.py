''' Python Standard Libraries '''
import os

''' Local Libraries '''
from helpers import debug, return_command_output
from env import (
        DGA_FILENAME, DGARCHIVE_FILENAME, DGARCHIVE_LINK, DGARCHIVE_USERNAME,
        DGARCHIVE_PASSWORD
        )

def parse_dga_line(line):
    ''' Parses DGArchive lines to extract the domain name '''

    line = line.strip()
    parts = line.split(',')
    name = parts[0].replace('"', '')

    return name

def load_dgarchive_names():
    ''' Loads DGArchive names to a set '''

    debug('Reading DGArchive and loading names to a set')
    all_dgas_set = set()

    with open(DGA_FILENAME) as infile:
        for line in infile:
            name = parse_dga_line(line)
            all_dgas_set.add(name)

    debug('DGA names have been loaded to a set')

    return all_dgas_set

def download_dgarchive():
    ''' Downloads the DGArchive dataset from the provided link '''

    debug('DGArchive dataset will be downloaded')

    os.chdir('aux_data')
    command = 'wget --user ' + str(DGARCHIVE_USERNAME) + ' --password ' \
            + str(DGARCHIVE_PASSWORD) + ' ' + str(DGARCHIVE_LINK)
    os.system(command)
    os.chdir('..')

    debug('The DGArchive dataset has been downloaded')

    return None

def untar_dataset_and_remove_p2p():
    ''' The DGArchive dataset will be decompressed and p2p files will be
        removed because we are interested only in DGA files
    '''

    debug('Untarring the downloaded DGArchive dataset')

    os.chdir('aux_data')
    command = 'tar -zxvf ' + str(DGARCHIVE_FILENAME)
    os.system(command)

    debug('The dataset has been untarred')

    debug('Erase p2p file because we want to keep the DGA names')
    os.system('rm ./*_p2p.csv')
    os.chdir('..')

    return None

def list_dga_files():
    ''' Lists all the available DGA files '''

    debug('Retrieve list of DGA files')
    
    command = 'ls aux_data/*_dga.csv'
    dga_files_terminal = return_command_output(command).decode('utf-8')
    dga_files_list = dga_files_terminal.split('\n')

    return dga_files_list

def determine_families(dga_files_list):
    ''' Parses the file names to extract the related DGA families '''

    dga_families_list = []

    for dga_filename in dga_files_list:
        temp = dga_filename.split('/')
        temp = temp[1]
        temp = temp.split('_')
        dga_name = temp[0]
        dga_families_list.append(dga_name)

    return dga_families_list

def sum_dga_files(dga_files_list):
    ''' All DGA files will be appended to a common file 
        Returns the names of the DGA families included in the DGArchive repo
    '''

    debug("Will append all DGA files to one")
    os.system('touch' + DGA_FILENAME)

    for dga_family in dga_files_list:
        command = 'cat ' + str(dga_family) + ' >> ' + DGA_FILENAME
        os.system(command)
        debug('Copied ' + str(dga_family) + ' to the full DGA file')

    return None

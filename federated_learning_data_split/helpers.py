''' Python Standard Libraries '''
import os
import shutil

''' Local Libraries '''
from env import DEBUG_SEPARATOR, DEBUG_ENABLED

def debug(*args):
    ''' Prints debug messages '''

    if DEBUG_ENABLED == True:
        for arg in args:
            print(arg)
    print(DEBUG_SEPARATOR, '\n')

    return None

def delete_file(filename):
    ''' Safely deletes the specified file '''

    if os.path.exists(filename):
        os.remove(filename)
        debug('Deleted file:', filename)

    return None

def delete_directory(directory_name):
    ''' Safely deletes directories '''

    if os.path.exists(directory_name):
        shutil.rmtree(directory_name, ignore_errors = True)
        debug('Deleted directory:', directory_name)

    return None

def create_directory(directory_name):
    ''' Created a directory '''

    os.mkdir(directory_name)
    debug('Created directory:', directory_name)

    return None

def del_move_to_directory(directory_name):
    ''' Moves to specific directory after deleting and recreating it '''

    delete_directory(directory_name)
    create_directory(directory_name)
    os.chdir(directory_name)

    return None

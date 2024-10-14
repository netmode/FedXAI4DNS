''' Python Standard Libraries '''
import os
import shutil
import subprocess

''' Local Libraries '''
from env import DEBUG_SEPARATOR, DEBUG_ENABLED

def debug(*args):
    ''' Prints debug messages '''

    if DEBUG_ENABLED == True:
        for arg in args:
            print(arg)
    print(DEBUG_SEPARATOR, '\n')

    return None

def return_command_output(command):
    ''' Returns the output of a Linux command '''
    proc = subprocess.Popen(command, stdout = subprocess.PIPE, shell = True)
    (out, err) = proc.communicate()
    output = out.rstrip('\n'.encode('utf8'))
    return output

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

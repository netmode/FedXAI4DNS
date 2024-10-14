import os
import subprocess

from env import CONSIDERED_FAMILIES, PARTICIPANTS, SHARED_FAMILY_RATIOS

def return_command_output(command):
    ''' Returns the output of a Linux command '''
    proc = subprocess.Popen(command, stdout = subprocess.PIPE, shell = True)
    (out, err) = proc.communicate()
    output = out.rstrip('\n'.encode('utf8'))
    return output

def inspect(prefix):
    for participant in range(PARTICIPANTS):
        filename = './' + str(prefix) + '/participant' + str(participant) + '.csv'
        print('Participant:', participant)
        for family in CONSIDERED_FAMILIES:
            command = 'cat ' + filename + ' | grep ,' + str(family) + ' | wc -l'
            output = return_command_output(command)
            print(family, output)
        print('-------------------------------------')

    return None

if __name__ == '__main__':
    print('Random Split')
    inspect('random_dga_split')

    print('Dedicated Split')
    inspect('dedicated_dga_split')

    for ratio in SHARED_FAMILY_RATIOS:
        print('Semirandom Split: ', ratio)
        prefix = 'semirandom_dga_split_' + str(ratio).replace('.', '')
        inspect(prefix)

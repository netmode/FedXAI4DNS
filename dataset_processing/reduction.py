from env import MAX_SIZE, TRANCO_SIZE, TRANCO_REMAINING_AFTER_TOP100K
from label import labeler
from dgarchive_processor import parse_dga_line

def reduce(line, prefix_set, suffixes):
    ''' Removes the public suffix from the given name 
        The argument is a line from the given file
    '''
    
    name = line.strip()
    labels = name.split('.')
    labels.reverse()
    candidate_suffix = labels[0]

    index = 0
    try:
        while candidate_suffix in suffixes:
            index += 1
            candidate_suffix = labels[index] + '.' + candidate_suffix
        labels.reverse()
        prefix = '.'.join(labels[0 : (len(labels)-index)])
        to_add = str(prefix) + ',' + str(name)
        prefix_set.add(to_add)
    except:
        pass

    return prefix_set

def reduce_tranco(suffixes):
    ''' Removes public suffixes from Tranco names '''

    filename = TRANCO_REMAINING_AFTER_TOP100K
    prefix_set = set()

    fdr = open(filename, 'r')

    for line in fdr:
        prefix_set = reduce(line, prefix_set, suffixes)
        if len(prefix_set) == TRANCO_SIZE:
            break

    fdr.close()

    return prefix_set

def reduce_dga(dga_family, suffixes):
    ''' Removes public suffixes from DGA names of a specific family '''

    filename = 'aux_data/' + str(dga_family) + '_dga.csv'
    prefix_set = set()

    fdr = open(filename, 'r')

    for line in fdr:
        line = parse_dga_line(line)
        prefix_set = reduce(line, prefix_set, suffixes)
        if len(prefix_set) == MAX_SIZE:
            break

    fdr.close()

    return prefix_set

def reduce_main(dga_families_list, suffixes, fdw):
    ''' The main process of removing public suffixes from Tranco and 
        DGA names 
    '''

    remaining_families = set()  # DGAs that will end up in the dataset
    total_sizes = []

    prefix_set = reduce_tranco(suffixes)
    remaining_families.add('tranco')
    for item in prefix_set:
        item = labeler(item, 'tranco')
        fdw.write(item + '\n')

    total_names = len(prefix_set)
    total_sizes.append(total_names)

    for dga_family in dga_families_list:
        prefix_set = reduce_dga(dga_family, suffixes)
        if len(prefix_set) < MAX_SIZE:
            continue
        remaining_families.add(dga_family)

        for item in prefix_set:
            item = labeler(item, dga_family)
            fdw.write(item + '\n')

        total_names = len(prefix_set)
        total_sizes.append(total_names)

    return remaining_families, total_sizes

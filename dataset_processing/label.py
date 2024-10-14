def labeler(item, family):
    ''' Assigns label to the respective item and adds the family name
        where item includes the extracted features and  the related name
        Binary classification only
    '''

    if family == 'tranco':
        item = item + ',0,' + str(family)
    else:
        item = item + ',1,' + str(family)

    return item

## FUTURE USE: For multiclass classification (not targeted in this work)
def assign_sequence_number(dga_families):
    ''' Assigns sequence numbers to Tranco and the DGA families '''

    dga_dict = {}
    dga_dict['tranco'] = 0

    sequence = 1
    for dga in dga_families:
        dga_dict[dga] = sequence
        sequence += 1

    return dga_dict

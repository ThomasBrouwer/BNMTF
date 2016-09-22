"""
Helper function for reading in the CCLE dataset, splitting into data X and mask M.
Returns:
    X               Drug sensitivity values (original)
    M               Mask of known vs unknown values
"""

import numpy, sys
sys.path.append("/home/thomas/Documents/Projects/libraries/")

folder_ccle = "/home/tab43/Documents/Projects/libraries/BNMTF/drug_sensitivity/data/ccle/"
ccle_ic_file = folder_ccle+"ic50.txt"
ccle_ec_file = folder_ccle+"ec50.txt"

def load_ccle(ic50=True, delim='\t'):
    filelocation = (ccle_ic_file if ic50 else ccle_ec_file)
    data = numpy.genfromtxt(filelocation, delimiter=delim, missing_values=[numpy.nan])
    I, J = data.shape
    
    # Construct the mask matrix, and replace any nan values by 0
    new_data, mask = numpy.zeros((I,J)), numpy.zeros((I,J))
    for i in range(0,I):
        for j in range(0,J):
            if not numpy.isnan(data[i,j]):
                new_data[i,j] = data[i,j]
                mask[i,j] = 1.
    return new_data, mask

'''
X, M = load_ccle(ic50=False)
(I,J)= X.shape
print I,J
print I*J, M.sum(), M.sum()/(I*J)
'''
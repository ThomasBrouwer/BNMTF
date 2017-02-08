"""
Helper function for reading in the Sanger dataset, splitting into data X,
mask M, drug names, cancer cell line names.
We exclude two lines from the dataset because on those cell lines only two 
drugs were tested.
Returns:
    X               Drug sensitivity values (original)
    X_min           Drug sensitivity values, minus (the lowest value in the dataset + 1)
    M               Mask of known vs unknown values
    drug_names      List of drug names
    cell_lines      List of which cell lines they are
    cancer_types    List of the cancer types of the cell lines
    tissues         List of tissue types of the cell lines
    
Also have a helper for storing it back into a file.
"""

import numpy

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
folder_gdsc = project_location+"BNMTF/data_drug_sensitivity/gdsc/"
gdsc_file = folder_gdsc+"ic50_excl_empty_filtered_cell_lines_drugs.txt"
gdsc_file_std = folder_gdsc+"ic50_excl_empty_filtered_cell_lines_drugs_standardised.txt"

def load_gdsc(location=None,standardised=False,sep=","):
    """ Load in data. We get a masked array, and set masked values to 0. """
    if location:
        fin = location
    else:
        fin = gdsc_file if not standardised else gdsc_file_std

    lines = [line.split("\n")[0].split("\r")[0].split(sep) for line in open(fin,'r').readlines()]
    drug_names = lines[0][3:]
    cell_lines = []
    cancer_types = []
    tissues = []
    X = []
    M = []
    for line in lines[1:]:
        cell_lines.append(line[0])
        cancer_types.append(line[1])
        tissues.append(line[2])
        X.append([float(v) if v != '' else 0.0 for v in line[3:]])
        M.append([1.0 if v != '' else 0.0 for v in line[3:]])
    X = numpy.array(X,dtype=float)
    M = numpy.array(M,dtype=float)
    
    minimum = X.min()-1
    X_min = []
    for row,row_M in zip(X,M):    
        X_min.append([v-minimum if m else 0.0 for v,m in zip(row,row_M)])
    X_min = numpy.array(X_min,dtype=float)
    
    return (X,X_min,M,drug_names,cell_lines,cancer_types,tissues)
    

def negate_gdsc(X,M):
    ''' Take in the Sanger dataset, take the negative of all values, and shift
        to positive values (+1), for interpretability. '''
    lowest_value = 0        
        
    X = -X
    minimum = X.min()-lowest_value
    
    X_min = []
    for row,row_M in zip(X,M):    
        X_min.append([v-minimum if m else 0.0 for v,m in zip(row,row_M)])
    X_min = numpy.array(X_min,dtype=float)
    
    return X_min
    
    
def store_gdsc(location,X,M,drug_names,cell_lines,cancer_types,tissues):
    ''' Store the data X. First line is drug names, then comes the data.
        For the data, first column is cell line name, second is cancer type, 
        third is tissue, then follows the drug sensitivity values.
        For missing values we store nothing '''
    fout = open(location,'w')
    fout.write("Cell Line\tCancer Type\tTissue\t" + "\t".join(drug_names) + "\n")
    
    for i,(cell_line,cancer_type,tissue,row) in enumerate(zip(cell_lines,cancer_types,tissues,X)):
        line = cell_line+"\t"+cancer_type+"\t"+tissue+"\t"
        data = [str(val) if M[i][j] else "" for (j,val) in enumerate(row)]
        line += "\t".join(data) + "\n"
        fout.write(line)
    fout.close()


def load_kernels(folder,file_names):
    ''' Load in all the files specified in the list <file_names> in <folder>,
        and return as a list, along with the drug/cell line names.'''
    kernels = []
    for name in file_names:
        lines = open(folder+name,'r').readlines()
        #entity_names = lines[0]
        values = [line.split("\t") for line in lines[1:]]
        kernel = numpy.array(values,dtype=float)
        kernels.append(kernel)
    return kernels
    
    
def load_features(location,delim="\t"):
    ''' Load in the features at the specified location, ignoring the first
        row (column names) and column (row names). '''
    lines = open(location,'r').readlines()
    lines = numpy.array([line.split("\n")[0].split(delim) for line in lines[1:]])
    values = numpy.array(lines[0:,1:],dtype=float)
    return (values)

'''
(X,X_min,M,drug_names,cell_lines,cancer_types,tissues) = load_gdsc()
(I,J)= X.shape
print I,J
print I*J, M.sum(), M.sum()/(I*J)
'''
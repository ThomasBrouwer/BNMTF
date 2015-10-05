"""
Methods for (randomly) generating a mask M of 1 values if a value is known, and
0 if a value is unknown
"""

import numpy, random, itertools

def generate_M(I,J,fraction):
    M = numpy.ones([I,J])
    values = random.sample(range(0,I*J),int(I*J*fraction))
    for v in values:
        M[v / J][v % J] = 0
    return M

# Compute <no_folds> folds, returning a list of M's. If M is defined, we split
# only the 1 entries into the folds.
def compute_folds(I,J,no_folds,M=None):
    if M is None:
        M = numpy.ones((I,J))
    else:
        M = numpy.array(M)
        
    no_elements = sum([len([v for v in row if v]) for row in M])
    indices = nonzero_indices(M)
    
    random.shuffle(indices)
    
    split_places = [int(i*no_elements/no_folds) for i in range(0,no_folds+1)] #find the indices where the next fold start
    split_indices = [indices[split_places[i]:split_places[i+1]] for i in range(0,no_folds)] #split the indices list into the folds
    
    folds_M = [] #list of the M's for the different folds
    for indices in split_indices:
        M = numpy.zeros((I,J))
        for i,j in indices:
            M[i][j] = 1
        folds_M.append(M)
    return folds_M
    
# Take in the ten fold M's, and construct the masks M for the other nine folds
def compute_Ms(folds_M):
    no_folds = len(folds_M)
    folds_M = [numpy.array(fold_M) for fold_M in folds_M]
    return [sum(folds_M[:fold]+folds_M[fold+1:]) for fold in range(0,no_folds)]

def calc_inverse_M(M):
    (I,J) = numpy.array(M).shape
    M_inv = numpy.ones([I,J])
    for i in range(0,I):
        for j in range(0,J):
            if M[i][j] == 1:
                M_inv[i][j] = 0
    return M_inv
    
# Return a list of indices of all nonzero indices in M
def nonzero_indices(M):
    (I,J) = numpy.array(M).shape
    return [(i,j) for i,j in itertools.product(range(0,I),range(0,J)) if M[i][j]]
    
# Return a list of lists, the ith list being of all indices j s.t. M[i,j] != 0
def nonzero_row_indices(M):
    (I,J) = numpy.array(M).shape
    return [[j for j,v in enumerate(row) if v] for row in M]
    
def nonzero_column_indices(M):
    M = numpy.array(M)
    (I,J) = numpy.array(M).shape
    return [[i for i,v in enumerate(column) if v] for column in M.T]

# Return a list of tuples of the actual value vs the predicted value, for nonzero elements in M
def recover_predictions(M,X_true,X_pred):
    (I,J) = numpy.array(M).shape
    actual_vs_predicted = []
    for i in range(0,I):
        for j in range(0,J):
            if M[i][j] == 0:
                actual_vs_predicted.append((X_true[i][j],X_pred[i][j]))
    return (actual_vs_predicted)
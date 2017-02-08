"""
Methods for (randomly) generating a mask M of 1 values if a value is known, and
0 if a value is unknown
"""

import numpy, random, itertools

# Generate a mask matrix M with <fraction> missing entries
def generate_M(I,J,fraction):
    M = numpy.ones([I,J])
    values = random.sample(range(0,I*J),int(I*J*fraction))
    for v in values:
        M[v / J][v % J] = 0
    return M
    
# Given a mask matrix M, generate an even more sparse matrix M_test, and M_train (s.t. M_test+M_train=M)
# The new mask matrix has <fraction> missing entries overall (so not fraction missing out of the observed entries, but out of all entries)
def generate_M_from_M(M,fraction):
    I,J = M.shape
    indices = nonzero_indices(M)
    no_elements = len(indices)
    no_missing_total = I*J - no_elements
    assert no_missing_total < I*J*fraction, "Specified %s fraction missing, so %s entries missing, but there are already %s missing by default!" % \
        (fraction,I*J*fraction,no_missing_total)
    
    # Shuffle the observed entries, take the first (I*J)*(1-fraction) and mark those as observed
    M_train, M_test = numpy.zeros((I,J)), numpy.zeros((I,J))
    random.shuffle(indices)
    index_last_observed = int(I*J*(1-fraction))
    
    for i,j in indices[:index_last_observed]:
        M_train[i,j] = 1
    for i,j in indices[index_last_observed:]:
        M_test[i,j] = 1
    assert numpy.array_equal(M,M_train+M_test), "Tried splitting M into M_test and M_train but something went wrong."
    return M_train, M_test
    
def try_generate_M_from_M(M,fraction,attempts):
    for i in range(0,attempts):
        M_train,M_test = generate_M_from_M(M,fraction)
        if check_empty_rows_columns(M_train):
            return M_train,M_test
    assert False, "Failed to generate folds for training and test data, %s attempts, fraction %s." % (attempts,fraction)

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
    
# Make n attempts to generate the folds with the training data having at least 1 observed entry per row and column
def compute_folds_attempts(I,J,no_folds,attempts,M=None):
    for i in range(0,attempts):
        folds_M = compute_folds(I=I,J=J,no_folds=no_folds,M=M)
        success = True
        for M_test in folds_M:
            M_train = M - M_test
            if not check_empty_rows_columns(M_train):
                success = False
        if success:
            return folds_M
    assert False, "Failed to generate folds for training and test data, %s attempts." % attempts
    
''' Make cross-validation folds, but only use the first amount of specified rows 
    or columns for the cross-validation splitting.
    Return a list of (train,test) matrices M. '''
def compute_crossval_folds_rows_attempts(M,no_rows,no_folds,attempts):
    I, J = M.shape
    M_rows = M[:no_rows]
    M_rest = M[no_rows:]  
        
    test_folds_M_rows = compute_folds_attempts(no_rows,J,no_folds,attempts,M_rows)
    
    train_folds_M, test_folds_M = [], []
    for test_fold_M_rows in test_folds_M_rows:
        # The train fold gets the rest of M, the test fold gets zeros there
        train_fold_M_rows = M_rows - test_fold_M_rows
        train_fold_M = numpy.concatenate((train_fold_M_rows,M_rest),axis=0)
        test_fold_M = numpy.concatenate((test_fold_M_rows,numpy.zeros((I-no_rows,J))),axis=0)
        
        train_folds_M.append(train_fold_M)
        test_folds_M.append(test_fold_M)
        
    return zip(train_folds_M,test_folds_M)
    
def compute_crossval_folds_columns_attempts(M,no_columns,no_folds,attempts):
    I, J = M.shape
    M_rows = M[:,:no_columns]
    M_rest = M[:,no_columns:]  
        
    test_folds_M_rows = compute_folds_attempts(I,no_columns,no_folds,attempts,M_rows)
    
    train_folds_M, test_folds_M = [], []
    for test_fold_M_rows in test_folds_M_rows:
        # The train fold gets the rest of M, the test fold gets zeros there
        train_fold_M_rows = M_rows - test_fold_M_rows
        train_fold_M = numpy.concatenate((train_fold_M_rows,M_rest),axis=1)
        test_fold_M = numpy.concatenate((test_fold_M_rows,numpy.zeros((I,J-no_columns))),axis=1)
        
        train_folds_M.append(train_fold_M)
        test_folds_M.append(test_fold_M)
        
    return zip(train_folds_M,test_folds_M)
    
# Return True if all rows and columns have at least one observation
def check_empty_rows_columns(M):
    sums_columns = M.sum(axis=0)
    sums_rows = M.sum(axis=1)
                
    # Assert none of the rows or columns are entirely unknown values
    for i,c in enumerate(sums_rows):
        if c == 0:
            return False
    for j,c in enumerate(sums_columns):
        if c == 0:
            return False
    return True
    
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
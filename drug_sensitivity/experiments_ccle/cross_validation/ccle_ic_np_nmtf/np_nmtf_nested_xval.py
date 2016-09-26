"""
Run the nested cross-validation for the NMTF class, on the CCLE IC50 dataset.
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")

import itertools
from BNMTF.code.nmtf_np import NMTF
from BNMTF.cross_validation.nested_matrix_cross_validation import MatrixNestedCrossValidation
from BNMTF.drug_sensitivity.experiments_ccle.load_data import load_ccle


# Settings
standardised = False
train_config = {
    'iterations' : 2000,
    'init_FG' : 'kmeans',
    'init_S' : 'exponential',
    'expo_prior' : 0.1
}
K_range = [1,2,3]
L_range = [1,2,3]
no_threads = 2
no_folds = 10
output_file = "./results.txt"
files_nested_performances = ["./fold_%s.txt" % fold for fold in range(1,no_folds+1)]

# Construct the parameter search
parameter_search = [{'K':K,'L':L} for (K,L) in itertools.product(K_range,L_range)]

# Load in the CCLE IC50 dataset
R,M = load_ccle(ic50=True)

# Run the cross-validation framework
#random.seed(42)
#numpy.random.seed(9000)
nested_crossval = MatrixNestedCrossValidation(
    method=NMTF,
    X=R,
    M=M,
    K=no_folds,
    P=no_threads,
    parameter_search=parameter_search,
    train_config=train_config,
    file_performance=output_file,
    files_nested_performances=files_nested_performances
)
nested_crossval.run()

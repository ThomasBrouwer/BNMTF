"""
Run the nested cross-validation for the NMF class, on the CCLE EC50 dataset.
"""

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.models.nmf_np import NMF
from BNMTF.code.cross_validation.nested_matrix_cross_validation import MatrixNestedCrossValidation
from BNMTF.experiments.experiments_ccle.load_data import load_ccle


# Settings
standardised = False
train_config = {
    'iterations' : 2000,
    'init_UV' : 'exponential',
    'expo_prior' : 0.1
}
K_range = [1,2,3]
no_threads = 5
no_folds = 10
output_file = "./results.txt"
files_nested_performances = ["./fold_%s.txt" % fold for fold in range(1,no_folds+1)]

# Construct the parameter search
parameter_search = [{'K':K} for K in K_range]

# Load in the CCLE EC50 dataset
R,M = load_ccle(ic50=False)

# Run the cross-validation framework
#random.seed(42)
#numpy.random.seed(9000)
nested_crossval = MatrixNestedCrossValidation(
    method=NMF,
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

"""
Run the cross validation with greedy search for model selection using VB-NMTF 
on the CCLE IC50 dataset.
"""

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

from BNMTF.code.models.bnmtf_vb_optimised import bnmtf_vb_optimised
from BNMTF.code.cross_validation.greedy_search_cross_validation import GreedySearchCrossValidation
from BNMTF.data_drug_sensitivity.ccle.load_data import load_ccle


# Settings
standardised = False
iterations = 1000

init_S = 'random' #'exp' #
init_FG = 'kmeans' #'exp' #

K_range = [4,5,6,7,8,9,10]
L_range = [4,5,6,7,8,9,10]
no_folds = 10
restarts = 1

quality_metric = 'AIC'
output_file = "./results.txt"

alpha, beta = 1., 1.
lambdaF = 1./10.
lambdaS = 1./10.
lambdaG = 1./10.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

# Load in the CCLE IC50 dataset
R,M = load_ccle(ic50=True)

# Run the cross-validation framework
#random.seed(42)
#numpy.random.seed(9000)
nested_crossval = GreedySearchCrossValidation(
    classifier=bnmtf_vb_optimised,
    R=R,
    M=M,
    values_K=K_range,
    values_L=L_range,
    folds=no_folds,
    priors=priors,
    init_S=init_S,
    init_FG=init_FG,
    iterations=iterations,
    restarts=restarts,
    quality_metric=quality_metric,
    file_performance=output_file
)
nested_crossval.run()

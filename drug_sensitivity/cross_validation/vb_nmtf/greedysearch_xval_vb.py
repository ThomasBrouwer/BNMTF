"""
Run the cross validation with greedy search for model selection using VB-NMTF 
on the Sanger dataset.
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")#("/home/thomas/Documenten/PhD/")#

import numpy, random
from BNMTF.code.bnmtf_vb_optimised import bnmtf_vb_optimised
from BNMTF.cross_validation.greedy_search_cross_validation import GreedySearchCrossValidation
from BNMTF.drug_sensitivity.load_data import load_Sanger


# Settings
standardised = False
iterations = 500

init_S = 'random' #'exp' #
init_FG = 'kmeans' #'exp' #

K_range = [4,5,6,7,8]
L_range = [4,5,6,7,8]
no_folds = 5
restarts = 2

quality_metric = 'AIC'
output_file = "./results.txt"

alpha, beta = 1., 1.
lambdaF = 1./10.
lambdaS = 1./10.
lambdaG = 1./10.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

# Load in the Sanger dataset
(_,X_min,M,_,_,_,_) = load_Sanger(standardised=standardised)

# Run the cross-validation framework
random.seed(42)
numpy.random.seed(9000)
nested_crossval = GreedySearchCrossValidation(
    classifier=bnmtf_vb_optimised,
    R=X_min,
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

"""

"""
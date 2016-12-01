"""
Run the cross validation with greedy search for model selection using VB-NMTF 
on the CCLE IC50 dataset.
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")#("/home/thomas/Documenten/PhD/")#

from BNMTF.code.nmtf_icm import nmtf_icm
from BNMTF.cross_validation.greedy_search_cross_validation import GreedySearchCrossValidation
from BNMTF.drug_sensitivity.experiments_ccle.load_data import load_ccle


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

minimum_TN = 0.01

# Load in the CCLE IC50 dataset
R,M = load_ccle(ic50=True)

# Run the cross-validation framework
#random.seed(1)
#numpy.random.seed(1)
nested_crossval = GreedySearchCrossValidation(
    classifier=nmtf_icm,
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
nested_crossval.run(minimum_TN=minimum_TN)

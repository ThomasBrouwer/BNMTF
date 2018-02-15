"""
Run the cross validation with line search for model selection using VB-NMF on
the CCLE IC50 dataset.
"""

import sys, os
project_location = os.path.dirname(__file__)+"/../../../../../"
sys.path.append(project_location)

from BNMTF.code.models.nmf_icm import nmf_icm
from BNMTF.code.cross_validation.line_search_cross_validation import LineSearchCrossValidation
from BNMTF.data_drug_sensitivity.ccle.load_data import load_ccle


# Settings
standardised = False
iterations = 1000
init_UV = 'random'

K_range = [1,2,3,4,5,6,7,8,9,10]
no_folds = 10
restarts = 1

quality_metric = 'AIC'
output_file = "./results.txt"

alpha, beta = 1., 1.
lambdaU = 1./10.
lambdaV = 1./10.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }

minimum_TN = 0.01

# Load in the CCLE IC50 dataset
R,M = load_ccle(ic50=True)

# Run the cross-validation framework
#random.seed(42)
#numpy.random.seed(9000)
nested_crossval = LineSearchCrossValidation(
    classifier=nmf_icm,
    R=R,
    M=M,
    values_K=K_range,
    folds=no_folds,
    priors=priors,
    init_UV=init_UV,
    iterations=iterations,
    restarts=restarts,
    quality_metric=quality_metric,
    file_performance=output_file
)
nested_crossval.run(minimum_TN=minimum_TN)

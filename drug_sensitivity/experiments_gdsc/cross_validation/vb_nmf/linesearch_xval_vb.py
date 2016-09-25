"""
Run the cross validation with line search for model selection using VB-NMF on
the Sanger dataset.
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")#("/home/thomas/Documenten/PhD/")#

import numpy, random
from BNMTF.code.bnmf_vb_optimised import bnmf_vb_optimised
from BNMTF.cross_validation.line_search_cross_validation import LineSearchCrossValidation
from BNMTF.drug_sensitivity.experiments_gdsc.load_data import load_gdsc


# Settings
standardised = False
iterations = 1000
init_UV = 'random'

K_range = [15,20,25,30]
no_folds = 10
restarts = 1

quality_metric = 'AIC'
output_file = "./REDO_results.txt"

alpha, beta = 1., 1.
lambdaU = 1./10.
lambdaV = 1./10.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaU':lambdaU, 'lambdaV':lambdaV }

# Load in the Sanger dataset
(_,X_min,M,_,_,_,_) = load_gdsc(standardised=standardised,sep=',')

# Run the cross-validation framework
#random.seed(42)
#numpy.random.seed(9000)
nested_crossval = LineSearchCrossValidation(
    classifier=bnmf_vb_optimised,
    R=X_min,
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
nested_crossval.run()

"""
all_MSE = [2.2242309355503416, 2.3108126630384804, 2.4095896447817631, 2.2188694213830114, 2.4185938516134278, 2.1808748510586002, 2.2503432196374651, 2.2305023229025145, 2.3595465204422488, 2.2186318302878667]
all_R2 = [0.8123419361488506, 0.8011409466575017, 0.7943028271877304, 0.8125046212085996, 0.7934881370166628, 0.8111969927756486, 0.8058878338360765, 0.811089129626958, 0.798953276136085, 0.8151865445946502]

Average MSE: 2.2821995260695718 +- 0.0066998949966021598
Average R^2: 0.80560922451887629 +- 5.8495363723835686e-05
"""
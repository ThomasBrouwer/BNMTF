"""
Run the cross validation with line search for model selection using VB-NMF on
the Sanger dataset.
"""

import sys
sys.path.append("/home/tab43/Documents/Projects/libraries/")#("/home/thomas/Documenten/PhD/")#

import numpy, random
from BNMTF.code.nmf_icm import nmf_icm
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

minimum_TN = 0.1

# Load in the Sanger dataset
(_,X_min,M,_,_,_,_) = load_gdsc(standardised=standardised)

# Run the cross-validation framework
#random.seed(42)
#numpy.random.seed(9000)
nested_crossval = LineSearchCrossValidation(
    classifier=nmf_icm,
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
nested_crossval.run(minimum_TN=minimum_TN)

"""
all_MSE = [3.5039148405029135, 9.0622730084824674, 3.7009069757338917, 3.3451246835265178, 3.1147595748400358, 3.9037354439533258, 13.991970030783968, 3.1814210224127897, 3.2677197491020404, 12.460551868851933]
all_R2 = [0.7072309782081623, 0.2162669348625822, 0.6853079551313846, 0.7144108917311998, 0.7341480430315861, 0.6671037956836574, -0.17013019643779437, 0.7288988508164431, 0.7201731755424339, -0.07478035943340289]

Average MSE: 5.953237719818989 +- 16.165927731904752
Average R^2: 0.49286300691362522 +- 0.11663176700952635
"""
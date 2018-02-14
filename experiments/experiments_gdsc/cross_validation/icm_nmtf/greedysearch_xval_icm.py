"""
Run the cross validation with greedy search for model selection using ICM-NMTF 
on the Sanger dataset.
"""

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

import numpy, random
from BNMTF.code.models.nmtf_icm import nmtf_icm
from BNMTF.code.cross_validation.greedy_search_cross_validation import GreedySearchCrossValidation
from BNMTF.data_drug_sensitivity.gdsc.load_data import load_gdsc


# Settings
standardised = False
iterations = 1000

init_S = 'random' #'exp' #
init_FG = 'kmeans' #'exp' #

K_range = [5,6,7,8,9,10]
L_range = [5,6,7,8,9,10]
no_folds = 10
restarts = 1

quality_metric = 'AIC'
output_file = "./results.txt"

alpha, beta = 1., 1.
lambdaF = 1./10.
lambdaS = 1./10.
lambdaG = 1./10.
priors = { 'alpha':alpha, 'beta':beta, 'lambdaF':lambdaF, 'lambdaS':lambdaS, 'lambdaG':lambdaG }

minimum_TN = 0.1

# Load in the Sanger dataset
(_,X_min,M,_,_,_,_) = load_gdsc(standardised=standardised,sep=',')

# Run the cross-validation framework
#random.seed(1)
#numpy.random.seed(1)
nested_crossval = GreedySearchCrossValidation(
    classifier=nmtf_icm,
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
nested_crossval.run(minimum_TN=minimum_TN)

"""
all_MSE = [2.2020002331612534, 2.2364503149918011, 2.1611831576199534, 2.1569381861635395, 2.1530470452271864, 2.272519698528658, 2.1910498022580613, 2.2302383199950797, 2.1027416628364484, 2.283196008129782]
all_R2 = [0.8068027775294401, 0.8122652321538621, 0.8155286993833876, 0.8151068635575036, 0.8227521825461013, 0.8062086302462692, 0.8136429679161671, 0.8113058601446024, 0.8152542609952846, 0.8080593057170452]

Average MSE: 2.1989364428911764 +- 0.0029521290510586768
Average R^2: 0.81269267801896627 +- 2.2283761452627026e-05
"""
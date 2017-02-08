"""
Run the cross validation with greedy search for model selection using VB-NMTF 
on the Sanger dataset.
"""

project_location = "/Users/thomasbrouwer/Documents/Projects/libraries/"
import sys
sys.path.append(project_location)

import numpy, random
from BNMTF.code.models.bnmtf_gibbs_optimised import bnmtf_gibbs_optimised
from BNMTF.code.cross_validation.greedy_search_cross_validation import GreedySearchCrossValidation
from BNMTF.experiments.experiments_gdsc.load_data import load_gdsc


# Settings
standardised = False
iterations = 1000
burn_in = 900
thinning = 2

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

# Load in the Sanger dataset
(_,X_min,M,_,_,_,_) = load_gdsc(standardised=standardised)

# Run the cross-validation framework
#random.seed(1)
#numpy.random.seed(1)
nested_crossval = GreedySearchCrossValidation(
    classifier=bnmtf_gibbs_optimised,
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
nested_crossval.run(burn_in=burn_in,thinning=thinning)

"""
all_MSE = [2.2840197244732074, 2.4010413568146909, 2.3867096829182866, 2.5140729100375911, 2.4161603588039613, 2.5768426948112859, 2.4258351325273564, 2.416620106102529, 2.2286332627076089, 2.3745461326347104]
all_R2 = [0.8033980427153291, 0.798845320492358, 0.8023608504542508, 0.7847220094659351, 0.7846794714863345, 0.7881485488273184, 0.7940181660135461, 0.7954596533423378, 0.8057721746024293, 0.7961801714226922]

Average MSE: 2.4024481361831223 +- 0.0089074472278596831
Average R^2: 0.79535844088225327 +- 5.1591154270217092e-05
"""